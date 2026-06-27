#!/bin/bash
# 本地 → 服务器：推送 Google Trends 数据和叙事到 mote-home
# 用法: bash sync_to_server.sh

SERVER="mote@100.118.10.0"
REMOTE_DIR="~/MemeticChaos"

echo "[sync_to_server] $(date '+%Y-%m-%d %H:%M')"

# Google Trends 数据 (本地飞鸟代理拉取)
echo "  → Google Trends data..."
scp data/collector/google_trends_2015_2025.json "$SERVER:$REMOTE_DIR/data/collector/"
scp data/collector/external_field_2015_2025.json "$SERVER:$REMOTE_DIR/data/collector/"

# 叙事 JSON
echo "  → Narratives..."
scp -r data/processed/narratives/ "$SERVER:$REMOTE_DIR/data/processed/"
scp -r data/processed/narratives_from_trends/ "$SERVER:$REMOTE_DIR/data/processed/"

# 新模型文件 (如有更新)
echo "  → Model files..."
scp src/models/order_form_predictor.py "$SERVER:$REMOTE_DIR/src/models/"
scp -r src/constraint/ "$SERVER:$REMOTE_DIR/src/"

echo "[sync_to_server] Done."
