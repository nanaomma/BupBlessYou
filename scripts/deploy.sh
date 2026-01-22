#!/bin/bash

set -e  # 에러 발생 시 스크립트 중단

echo "🚀 Starting deployment..."

# 프로젝트 디렉토리
PROJECT_DIR=~/BupBlessYou
cd $PROJECT_DIR

# 1. 환경 변수 로드
echo "📋 Loading environment variables..."
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  Warning: .env file not found. Using environment defaults."
fi

# 2. Python 가상환경 활성화
echo "🐍 Activating Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 3. 의존성 설치
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. 데이터베이스 마이그레이션
echo "🗄️  Running database migrations..."
alembic upgrade head

# 5. 기존 프로세스 종료
echo "🛑 Stopping existing services..."

# FastAPI 프로세스 종료
if [ -f fastapi.pid ]; then
    PID=$(cat fastapi.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping FastAPI (PID: $PID)..."
        kill $PID
        sleep 2
        # 강제 종료
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
    fi
    rm fastapi.pid
fi

# 6. 로그 디렉토리 생성
echo "📂 Creating log directory..."
mkdir -p logs

# 7. FastAPI 서버 시작
echo "🚀 Starting FastAPI server..."
nohup uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    > logs/fastapi.log 2>&1 &
echo $! > fastapi.pid
echo "✅ FastAPI started (PID: $(cat fastapi.pid))"

echo "🎉 Deployment completed successfully!"
echo "📊 FastAPI: http://$(hostname -I | awk '{print $1}'):8000"
echo "📝 Logs: $PROJECT_DIR/logs/"
