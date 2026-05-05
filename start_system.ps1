Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "🚀 HELLO WORLD: INITIALIZING MLOPS INFRASTRUCTURE" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Cleaning up old containers
Write-Host "🧹 Step 1: Cleaning up existing host resources..." -ForegroundColor Yellow
docker-compose down --remove-orphans

# 2. Building Docker Images
Write-Host "🏗️ Step 2: Building Microservices (Backend & Frontend)..." -ForegroundColor Yellow
docker-compose build

# 3. Initializing Host Network and Starting Services
Write-Host "🌐 Step 3: Launching services on Host Network..." -ForegroundColor Yellow
docker-compose up -d

# --- NEW: Waiting for Backend to Load Models ---
Write-Host "⏳ Step 4: Waiting for AI Backend to load models (this takes ~60s)..." -ForegroundColor Cyan
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/docs" -Method Get -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "🚀 Backend is ONLINE and ready!" -ForegroundColor Green
            break
        }
    } catch {}
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 5
}

# 5. Final Status Check
Write-Host "`n✅ System Initialized Successfully!" -ForegroundColor Green
Write-Host "-----------------------------------------------"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host "-----------------------------------------------"
Write-Host "👉 Web App: http://localhost:8501"
Write-Host "👉 Backend API: http://localhost:8000"
Write-Host "👉 MLflow: http://localhost:5000"
Write-Host "==============================================="
