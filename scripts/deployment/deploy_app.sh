#!/bin/bash

# Application Deployment Script for Google Trends Quantile Forecaster
# This script deploys the application using Docker on EC2

set -e  # Exit on any error

# Configuration
APP_NAME="google-trends-forecaster"
APP_DIR="/opt/google-trends-forecaster"
LOG_DIR="/var/log/google-trends-forecaster"
DOCKER_IMAGE_NAME="google-trends-forecaster"
DOCKER_CONTAINER_NAME="google-trends-forecaster-app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
   exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    error "Docker is not running. Please start Docker first."
    exit 1
fi

# Function to stop and remove existing container
stop_existing_container() {
    log "Checking for existing container..."
    if docker ps -a --format "table {{.Names}}" | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        log "Stopping existing container..."
        docker stop ${DOCKER_CONTAINER_NAME} || true
        log "Removing existing container..."
        docker rm ${DOCKER_CONTAINER_NAME} || true
    fi
}

# Function to remove old images
cleanup_old_images() {
    log "Cleaning up old images..."
    docker image prune -f || true
}

# Function to build new image
build_image() {
    log "Building Docker image..."
    cd ${APP_DIR}
    
    if [ ! -f "Dockerfile" ]; then
        error "Dockerfile not found in ${APP_DIR}"
        exit 1
    fi
    
    docker build -t ${DOCKER_IMAGE_NAME}:latest .
    
    if [ $? -eq 0 ]; then
        log "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run container
run_container() {
    log "Starting application container..."
    
    # Create necessary directories if they don't exist
    mkdir -p ${APP_DIR}/data
    mkdir -p ${APP_DIR}/models
    mkdir -p ${APP_DIR}/logs
    mkdir -p ${APP_DIR}/mlruns
    
    # Run the container
    docker run -d \
        --name ${DOCKER_CONTAINER_NAME} \
        --restart unless-stopped \
        -p 5000:5000 \
        -v ${APP_DIR}/data:/app/data \
        -v ${APP_DIR}/models:/app/models \
        -v ${APP_DIR}/logs:/app/logs \
        -v ${APP_DIR}/mlruns:/app/mlruns \
        -v ${LOG_DIR}:/var/log/google-trends-forecaster \
        --env-file ${APP_DIR}/.env \
        ${DOCKER_IMAGE_NAME}:latest
    
    if [ $? -eq 0 ]; then
        log "Container started successfully"
    else
        error "Failed to start container"
        exit 1
    fi
}

# Function to check container health
check_health() {
    log "Checking container health..."
    
    # Wait for container to start
    sleep 10
    
    # Check if container is running
    if docker ps --format "table {{.Names}}" | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        log "Container is running"
    else
        error "Container is not running"
        docker logs ${DOCKER_CONTAINER_NAME}
        exit 1
    fi
    
    # Check health endpoint
    log "Checking application health..."
    for i in {1..30}; do
        if curl -f http://localhost:5000/health > /dev/null 2>&1; then
            log "Application is healthy and responding"
            return 0
        fi
        log "Waiting for application to be ready... (attempt $i/30)"
        sleep 5
    done
    
    error "Application health check failed"
    docker logs ${DOCKER_CONTAINER_NAME}
    exit 1
}

# Function to show deployment status
show_status() {
    log "Deployment completed successfully!"
    log "Application is running on http://localhost:5000"
    log "Container name: ${DOCKER_CONTAINER_NAME}"
    log "To view logs: docker logs ${DOCKER_CONTAINER_NAME}"
    log "To stop application: docker stop ${DOCKER_CONTAINER_NAME}"
}

# Main deployment process
main() {
    log "Starting deployment of ${APP_NAME}..."
    
    # Check if app directory exists
    if [ ! -d "${APP_DIR}" ]; then
        error "Application directory ${APP_DIR} does not exist"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "${APP_DIR}/.env" ]; then
        warning ".env file not found. Using default environment variables."
        # Create a basic .env file if it doesn't exist
        cat > ${APP_DIR}/.env << EOF
FLASK_ENV=production
SECRET_KEY=change-this-in-production
API_VERSION=v1
API_TITLE=Google Trends Quantile Forecaster API
API_DESCRIPTION=API for forecasting Google Trends data using LSTM models
RATE_LIMIT_DEFAULT=50/hour
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=google_trends_forecaster
DATA_DIR=data
MODELS_DIR=models
LOGS_DIR=logs
PYTRENDS_DELAY=1
PYTRENDS_RETRIES=3
PYTRENDS_TIMEOUT=30
DEFAULT_BATCH_SIZE=5
DEFAULT_EPOCHS=150
DEFAULT_PREDICTION_WEEKS=25
DEFAULT_LSTM_UNITS=4
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=5000
EOF
    fi
    
    stop_existing_container
    cleanup_old_images
    build_image
    run_container
    check_health
    show_status
}

# Run main function
main "$@" 