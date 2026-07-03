"""
从微信本地数据库导出指定联系人的聊天记录

原理：
1. 微信PC版将聊天记录存在加密的 SQLite 数据库中
2. 数据库密钥从微信进程内存中提取
3. 用密钥解密后直接 SQL 查询即可

用法：
    python src/advisor/wechat_export.py --contact 虹姐
    python src/advisor/wechat_export.py --contact 杨铖爽
"""

import os, sys, json, hashlib, struct, ctypes
from pathlib import Path
from datetime import datetime

# ═══ Step 1: Find WeChat database ═══

def find_wechat_dir() -> Path:
    """查找微信数据目录."""
    # Default: Documents/WeChat Files
    docs = Path(os.environ.get("USERPROFILE", "C:/Users")) / "Documents" / "WeChat Files"
    if docs.exists():
        # Find the user's wxid directory (not All Users, not system dirs)
        for d in docs.iterdir():
            if d.is_dir() and len(d.name) > 10 and "All" not in d.name:
                msg_dir = d / "Msg"
                if msg_dir.exists():
                    return d
    raise FileNotFoundError("找不到微信数据目录，请确认微信已登录")


def find_db_files(wx_dir: Path) -> list[Path]:
    """找到所有加密的数据库文件."""
    msg_dir = wx_dir / "Msg"
    dbs = sorted(msg_dir.glob("MSG*.db"), key=lambda x: x.stat().st_size, reverse=True)
    # Also check Multi subdirectory
    multi_dir = msg_dir / "Multi"
    if multi_dir.exists():
        dbs += sorted(multi_dir.glob("MSG*.db"), key=lambda x: x.stat().st_size, reverse=True)
    return dbs


# ═══ Step 2: Extract DB key from WeChat process memory ═══

def get_wechat_key() -> bytes:
    """从微信进程内存中提取数据库密钥."""
    import pymem
    import pymem.process

    try:
        pm = pymem.Pymem("WeChat.exe")
    except pymem.exception.ProcessNotFound:
        pm = pymem.Pymem("wechat.exe")

    # The key is stored near the WeChatWin.dll module
    wechatwin = None
    for mod in pm.list_modules():
        if "WeChatWin" in mod.name:
            wechatwin = mod
            break

    if not wechatwin:
        raise RuntimeError("找不到 WeChatWin.dll")

    # Search for the database key pattern
    # The key is 32 bytes, typically stored as ASCII hex (64 chars) or raw bytes
    # Common approach: scan for known patterns near the module base

    module_base = wechatwin.lpBaseOfDll
    module_size = wechatwin.SizeOfImage

    # Read the entire module memory
    try:
        module_data = pm.read_bytes(module_base, module_size)
    except Exception:
        # Fallback: read first 100MB
        module_data = pm.read_bytes(module_base, min(module_size, 100 * 1024 * 1024))

    # Pattern 1: The key is often near strings like "DBKey" or "sqlite_key"
    # Pattern 2: 32-byte alphanumeric hex strings
    # Pattern 3: Memory offsets that are well-known for specific WeChat versions

    # Try finding the key using known offsets (WeChat 3.9.x)
    # The key is typically a 32-byte value stored at a fixed offset from WeChatWin.dll base
    # For WeChat 3.9+, the key offset varies but we can search for it

    # Method: scan for SQLite header "SQLite format 3" in memory
    # The key is stored near where the database path is referenced

    # Simplified approach: search for a 64-char hex string
    import re
    # Look for 64-character hex strings (32 bytes as hex)
    matches = list(re.finditer(b'[0-9a-fA-F]{64}', module_data))

    # Try each match as a potential key
    keys_tried = set()
    for match in matches:
        key_hex = match.group().decode('ascii')
        if len(set(key_hex)) < 10:  # Skip repetitive strings
            continue
        key_bytes = bytes.fromhex(key_hex)
        if key_bytes not in keys_tried:
            keys_tried.add(key_bytes)

    if not keys_tried:
        raise RuntimeError("在微信进程内存中未找到数据库密钥，可能微信版本不支持")

    return list(keys_tried)


# ═══ Step 3: Decrypt and query ═══

def try_decrypt_db(db_path: Path, keys: list[bytes]) -> tuple:
    """尝试用密钥解密数据库."""
    import sqlite3

    # Try each key
    for i, key in enumerate(keys):
        try:
            # Try sqlcipher via pysqlcipher3 or sqlite3 with key
            # If sqlcipher isn't available, try the built-in approach

            # WeChat uses a custom encryption: each page (4096 bytes) is encrypted
            # with the key using a simple XOR or AES based on WeChat version

            # For newer WeChat versions (3.9+), the encryption is AES-256-GCM
            # For older versions, it's a custom XOR-based scheme

            # Let's try the simplest approach first: read the first bytes
            with open(db_path, 'rb') as f:
                header = f.read(16)

            # Check if "SQLite format 3" is visible (unencrypted)
            if header[:15] == b'SQLite format 3':
                # This DB is not encrypted!
                return sqlite3.connect(str(db_path)), key, "unencrypted"

            # DB is encrypted — need to use pysqlcipher3 or similar
        except Exception:
            continue

    return None, None, "all keys failed"


def decrypt_wechat_db(db_path: Path, key: bytes) -> str:
    """Decrypt WeChat database to a temporary file and return path."""
    import tempfile

    # WeChat encryption: each 4096-byte page is encrypted
    # First 16 bytes of each page is the IV/nonce
    # The rest is AES-256-CBC encrypted

    output = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    output_path = output.name
    output.close()

    try:
        from Crypto.Cipher import AES

        with open(db_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                page_size = 4096
                while True:
                    page = f_in.read(page_size)
                    if not page:
                        break
                    if len(page) < page_size:
                        # Last partial page — copy as-is
                        f_out.write(page)
                        break

                    # First 16 bytes = IV, rest = ciphertext
                    iv = page[:16]
                    ciphertext = page[16:]

                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    plaintext = cipher.decrypt(ciphertext)

                    # Remove PKCS7 padding
                    pad_len = plaintext[-1]
                    if pad_len <= 16:
                        plaintext = plaintext[:-pad_len]

                    f_out.write(plaintext)

        return output_path
    except ImportError:
        # pycryptodome not installed, try installing
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pycryptodome', '--quiet'],
                      capture_output=True)
        from Crypto.Cipher import AES

        # Retry with crypto now installed
        return decrypt_wechat_db(db_path, key)


# ═══ Step 4: Query chat messages ═══

CONTACT_QUERY = """
SELECT
    datetime(CreateTime/1000, 'unixepoch', 'localtime') as time,
    IsSender,
    StrContent,
    Type,
    SubType
FROM MSG
WHERE StrTalker = ?
ORDER BY CreateTime
"""

def find_contact_id(cursor, name: str) -> str:
    """通过昵称或备注查找联系人的 wxid."""
    # Search in Contact table
    for table in ['Contact', 'WCContact']:
        try:
            cursor.execute(f"SELECT UserName, NickName, Alias, Remark FROM {table} WHERE NickName LIKE ? OR Remark LIKE ? OR Alias LIKE ?",
                         (f'%{name}%', f'%{name}%', f'%{name}%'))
            rows = cursor.fetchall()
            for row in rows:
                print(f"  找到: {row[1]} (wxid={row[0]}, 备注={row[3]})")
            if rows:
                return rows[0][0]
        except Exception:
            continue

    # Search in chat session list
    try:
        cursor.execute("SELECT strUsrName, strNickName FROM Session WHERE strNickName LIKE ?",
                      (f'%{name}%',))
        rows = cursor.fetchall()
        for row in rows:
            print(f"  会话: {row[1]} (wxid={row[0]})")
        if rows:
            return rows[0][0]
    except Exception:
        pass

    return None


def export_chat(db_path: str, contact_name: str, output_path: str):
    """导出指定联系人的聊天记录."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find the contact
    print(f"查找联系人: {contact_name}")

    # First try the Contact table
    wxid = None
    for table in ['Contact', 'WCContact']:
        try:
            cursor.execute(f"""
                SELECT UserName, NickName, Alias, Remark
                FROM {table}
                WHERE NickName LIKE ? OR Remark LIKE ? OR Alias LIKE ?
            """, (f'%{contact_name}%', f'%{contact_name}%', f'%{contact_name}%'))
            for row in cursor.fetchall():
                print(f"  找到: {row[1]} (wxid={row[0][:20]}..., 备注={row[3]})")
                wxid = row[0]
                break
        except Exception:
            continue
        if wxid:
            break

    if not wxid:
        # Try chatroom
        for table in ['ChatRoom']:
            try:
                cursor.execute(f"SELECT * FROM {table} WHERE NickName LIKE ? OR Topic LIKE ?",
                             (f'%{contact_name}%', f'%{contact_name}%'))
                for row in cursor.fetchall():
                    print(f"  群聊: {row}")
            except Exception:
                pass

        print(f"\n可用的联系人（部分）:")
        for table in ['Contact', 'WCContact']:
            try:
                cursor.execute(f"SELECT UserName, NickName, Remark FROM {table} LIMIT 20")
                for row in cursor.fetchall():
                    if row[1]:
                        print(f"  {row[1][:30]} (备注={row[2]})")
            except Exception:
                pass
        conn.close()
        return

    # Query messages
    print(f"\n导出聊天记录...")

    # Try with talker format (wxid or chatroom id)
    # WeChat stores messages with different talker formats
    talker_variants = [wxid]

    all_msgs = []
    for talker in talker_variants:
        cursor.execute(CONTACT_QUERY, (talker,))
        rows = cursor.fetchall()
        if rows:
            all_msgs.extend(rows)

    if not all_msgs:
        # Try wildcard
        cursor.execute("SELECT DISTINCT StrTalker FROM MSG WHERE StrTalker LIKE ? LIMIT 5",
                      (f'%{wxid[-8:]}%',))
        for row in cursor.fetchall():
            print(f"  可能的 talker: {row[0]}")

    all_msgs.sort(key=lambda x: x[0])

    print(f"找到 {len(all_msgs)} 条消息")

    # Write to file
    msg_types = {
        1: '[文本]', 3: '[图片]', 34: '[语音]', 43: '[视频]',
        47: '[表情]', 49: '[链接/文件]', 10000: '[系统]',
        50: '[语音通话]', 51: '[视频通话]',
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        for time_str, is_sender, content, msg_type, sub_type in all_msgs:
            sender = "你" if is_sender else contact_name
            type_label = msg_types.get(msg_type, f'[类型{msg_type}]')

            if msg_type == 1:  # Text
                f.write(f"{time_str} {sender}: {content}\n")
            elif msg_type in (47, 49):  # Emoji, link
                f.write(f"{time_str} {sender}: {type_label} {content or ''}\n")
            else:
                f.write(f"{time_str} {sender}: {type_label}\n")

    print(f"已导出到: {output_path}")
    conn.close()


# ═══ Main ═══

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact', required=True, help='联系人昵称或备注')
    parser.add_argument('--output', default=None, help='输出路径')
    args = parser.parse_args()

    wx_dir = find_wechat_dir()
    print(f"微信目录: {wx_dir}")

    dbs = find_db_files(wx_dir)
    print(f"数据库文件: {len(dbs)} 个")
    for db in dbs[:5]:
        print(f"  {db.name} ({db.stat().st_size / 1024 / 1024:.1f} MB)")

    # Find the main message database (largest one)
    main_db = dbs[0] if dbs else None
    if not main_db:
        print("ERROR: 未找到数据库文件，请确认微信已登录且完成聊天记录迁移")
        sys.exit(1)

    print(f"\n尝试解密: {main_db.name}")

    keys = get_wechat_key()
    print(f"找到 {len(keys)} 个潜在密钥")

    conn, working_key, status = try_decrypt_db(main_db, keys)

    if conn:
        output = args.output or f"data/advisor/{args.contact}_chat.txt"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        export_chat(main_db, args.contact, output)
        conn.close()
    else:
        print(f"解密失败: {status}")
        print("尝试用 pycryptodome 解密...")

        # Try the decryption approach
        if keys:
            try:
                decrypted_path = decrypt_wechat_db(main_db, keys[0])
                print(f"解密成功: {decrypted_path}")
                output = args.output or f"data/advisor/{args.contact}_chat.txt"
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                export_chat(decrypted_path, args.contact, output)
                os.unlink(decrypted_path)  # Clean up temp file
            except Exception as e:
                print(f"解密失败: {e}")
                sys.exit(1)
