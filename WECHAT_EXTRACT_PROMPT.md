# WeChatOffload — 微信聊天记录提取 提示词

## 目标

从电脑微信提取指定联系人和群聊的**完整**聊天记录，导出为结构化文本。

**需要导出**：
- 联系人：虹姐（私聊）
- 群聊：徐子浩服务指导1.29-2.28

**完整**意味着：
- 每条消息标注发送者（群聊中区分不同人，私聊中区分"我"和"对方"）
- 每条消息标注精确时间（至少到分钟，能到秒更好）
- 文本消息保留原文
- 图片标注为 `[图片]`（不丢）
- 微信自带小表情（如 /Doge）保留原文，不额外标注
- 自定义动画表情包标注为 `[动画表情]`
- 语音/视频通话标注为 `[语音通话]` 或 `[视频通话]`
- 文件标注为 `[文件: 文件名]`
- 其他消息类型（红包、转账、位置、名片等）至少标注类型
- 不截断、不省略、不合并相邻消息

## 环境

- Windows 11 Pro
- 微信版本：**4.1.10.53**（进程名 **Weixin.exe**，不是 WeChat.exe）
- `C:\Program Files\Tencent\Weixin\Weixin.exe`
- Python 环境：conda `MemeticChaos`，路径 `C:\anaconda3\envs\MemeticChaos\`

## 微信数据磁盘布局

### 当前活跃账号

wxid：`wxid_n6d1i0eeukms12`（ChatMsg.db 今天仍在更新，确认是活跃账号）

```
旧格式路径 (WeChat 3.x，仍在被 4.x 写入)：
  D:\soft\WeChat Files\wxid_n6d1i0eeukms12\Msg\
    ├── ChatMsg.db       96KB   (壳数据库，今天更新)
    ├── MicroMsg.db      224KB  (已确认加密)
    ├── Multi/
    │   ├── MSG0.db      60MB   ★ 真正的聊天数据库
    │   ├── MSG0.db-wal  3.8MB  (写前日志)
    │   ├── bak/         114MB  (备份)
    │   └── config.ini          (内容仅一行：MSG0.db)
    └── ...

新格式路径 (WeChat 4.x)：
  D:\soft\xwechat_files\wxid_n6d1i0eeukms12_6e10\
    ├── msg/             29MB   (附件/文件/视频目录)
    ├── db_storage/
    │   ├── message/     3.4MB  (消息数据库文件)
    │   ├── contact/     1.4MB
    │   ├── session/     428KB
    │   └── ...
    ├── config/
    │   ├── login_config      4096B (全零)
    │   └── login_configv2    4096B (二进制数据，疑似密钥材料)
    └── cache/           3.5MB
```

### 另一个旧账号（非当前）

wxid：`wxid_n9z5qzgrufno12`（ChatMsg.db 最后更新 2025年6月，已废弃）

## 已知事实

### 磁盘数据

- 数据库全部加密，密钥在 Weixin.dll (176MB) 进程内存中
- MSG0.db 头部 16 字节全零，17-32 字节为密文 salt
- 加密方案：AES-256-GCM，PBKDF2-HMAC-SHA512，256000 轮
- login_configv2 (4096B) 含二进制密钥材料，非 DPAPI 加密

### 进程信息

- Weixin.exe PID 约 23040
- Weixin.dll 基址约 0x7ffcd9700000，176MB

## 不要做

- 不要用截图/OCR/滚动——丢信息、分不清角色
- 不要用 pywinauto Ctrl+A Ctrl+C——同样的问题

## 输出要求

- 两个 txt 文件，每条消息格式：`[时间] 发送者: 内容`
- 时间精确到分钟
- 区分发送者
- 表情包标注为 `[表情]` 即可，不需要渲染

## 你自己探索

以上是已知信息。具体怎么做——你来决定。不要被上面的失败列表限制思路。
