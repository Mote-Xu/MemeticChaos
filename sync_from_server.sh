#!/bin/bash
# 从 mote-home 同步最新数据到本地
# 服务器是项目主阵地，本地用这个脚本拉取结果
SERVER="mote@100.118.10.0"
REMOTE="~/MemeticChaos"

echo "=== MemeticChaos 数据同步 ==="
echo "[1/5] 微博采集数据..."
scp -r "$SERVER:$REMOTE/data/collector/" data/collector/ 2>/dev/null
echo "[2/5] 轨迹更新..."
scp "$SERVER:$REMOTE/data/processed/trajectories.json" data/processed/ 2>/dev/null
echo "[3/5] 预测报告..."
scp "$SERVER:$REMOTE/data/processed/order_form_report.txt" data/processed/ 2>/dev/null
echo "[4/5] 日志..."
scp "$SERVER:$REMOTE/data/collector/cron_*.log" data/collector/ 2>/dev/null
echo "[5/5] 微博采集数据..."
scp -r "$SERVER:$REMOTE/data/scraped/" data/scraped/ 2>/dev/null
echo ""
echo "=== 同步完成 ==="
echo "运行 'python src/meme_inspector.py --list' 查看最新状态"
echo "运行 'python src/analysis/collective_dynamics.py' 生成最新相图"
echo "运行 'cat data/processed/order_form_report.txt' 查看最新预测报告"
