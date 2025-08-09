#!/bin/bash

# Health Check Script for Google Trends Quantile Forecaster
# This script monitors the application health and can be used for monitoring/alerting

set -e

# Configuration
APP_NAME="google-trends-forecaster"
DOCKER_CONTAINER_NAME="google-trends-forecaster-app"
HEALTH_URL="http://localhost:5000/health"
LOG_FILE="/var/log/google-trends-forecaster/health_check.log"
ALERT_EMAIL=""  # Set this to your email for alerts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a ${LOG_FILE}
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a ${LOG_FILE}
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a ${LOG_FILE}
}

# Function to check if Docker container is running
check_container_status() {
    if docker ps --format "table {{.Names}}" | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        return 0
    else
        return 1
    fi
}

# Function to check application health endpoint
check_health_endpoint() {
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" ${HEALTH_URL} 2>/dev/null || echo "000")
    if [ "$response_code" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check container resource usage
check_resource_usage() {
    local container_stats=$(docker stats ${DOCKER_CONTAINER_NAME} --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "N/A")
    echo "$container_stats"
}

# Function to check disk usage
check_disk_usage() {
    local disk_usage=$(df -h /opt/google-trends-forecaster | tail -1 | awk '{print $5}' | sed 's/%//')
    echo "$disk_usage"
}

# Function to send alert (placeholder - implement your preferred alerting method)
send_alert() {
    local message="$1"
    local subject="[ALERT] ${APP_NAME} Health Check Failed"
    
    # Log the alert
    error "ALERT: $message"
    
    # If email is configured, send alert
    if [ -n "$ALERT_EMAIL" ] && command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    fi
    
    # You can add other alerting methods here (Slack, PagerDuty, etc.)
    # Example for Slack webhook:
    # if [ -n "$SLACK_WEBHOOK_URL" ]; then
    #     curl -X POST -H 'Content-type: application/json' \
    #          --data "{\"text\":\"$message\"}" \
    #          "$SLACK_WEBHOOK_URL"
    # fi
}

# Function to restart application
restart_application() {
    warning "Attempting to restart application..."
    
    # Stop container
    docker stop ${DOCKER_CONTAINER_NAME} || true
    
    # Wait a moment
    sleep 5
    
    # Start container
    docker start ${DOCKER_CONTAINER_NAME} || true
    
    # Wait for startup
    sleep 10
    
    # Check if restart was successful
    if check_container_status && check_health_endpoint; then
        log "Application restart successful"
        return 0
    else
        error "Application restart failed"
        return 1
    fi
}

# Function to perform comprehensive health check
perform_health_check() {
    local status="healthy"
    local issues=()
    
    log "Starting health check for ${APP_NAME}..."
    
    # Check 1: Container status
    if ! check_container_status; then
        status="unhealthy"
        issues+=("Container is not running")
    else
        log "✓ Container is running"
    fi
    
    # Check 2: Health endpoint
    if ! check_health_endpoint; then
        status="unhealthy"
        issues+=("Health endpoint is not responding")
    else
        log "✓ Health endpoint is responding"
    fi
    
    # Check 3: Resource usage
    local resource_usage=$(check_resource_usage)
    if [ "$resource_usage" != "N/A" ]; then
        log "✓ Resource usage: $resource_usage"
    else
        warning "Could not retrieve resource usage"
    fi
    
    # Check 4: Disk usage
    local disk_usage=$(check_disk_usage)
    if [ -n "$disk_usage" ] && [ "$disk_usage" -gt 80 ]; then
        status="warning"
        issues+=("Disk usage is high: ${disk_usage}%")
        warning "High disk usage: ${disk_usage}%"
    else
        log "✓ Disk usage is normal: ${disk_usage}%"
    fi
    
    # Check 5: Log file size
    if [ -f "$LOG_FILE" ]; then
        local log_size=$(du -h "$LOG_FILE" | cut -f1)
        log "✓ Log file size: $log_size"
    fi
    
    # Report status
    if [ "$status" = "healthy" ]; then
        log "Health check passed - Application is healthy"
        return 0
    elif [ "$status" = "warning" ]; then
        warning "Health check warning - ${issues[*]}"
        return 1
    else
        error "Health check failed - ${issues[*]}"
        
        # Attempt restart if container is down
        if ! check_container_status; then
            warning "Attempting automatic restart..."
            if restart_application; then
                log "Automatic restart successful"
                return 0
            else
                send_alert "Application is down and automatic restart failed. Manual intervention required."
                return 1
            fi
        else
            send_alert "Application health check failed: ${issues[*]}"
            return 1
        fi
    fi
}

# Function to show detailed status
show_detailed_status() {
    log "=== Detailed Application Status ==="
    
    # Container status
    if check_container_status; then
        log "Container Status: RUNNING"
        docker ps --filter "name=${DOCKER_CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        error "Container Status: NOT RUNNING"
    fi
    
    # Container logs (last 10 lines)
    log "=== Recent Container Logs ==="
    docker logs --tail 10 ${DOCKER_CONTAINER_NAME} 2>/dev/null || error "Could not retrieve container logs"
    
    # Resource usage
    log "=== Resource Usage ==="
    check_resource_usage
    
    # Disk usage
    log "=== Disk Usage ==="
    df -h /opt/google-trends-forecaster
    
    # Health endpoint response
    log "=== Health Endpoint Response ==="
    curl -s ${HEALTH_URL} 2>/dev/null | head -5 || error "Health endpoint not responding"
}

# Main function
main() {
    # Create log directory if it doesn't exist
    mkdir -p $(dirname ${LOG_FILE})
    
    # Parse command line arguments
    case "${1:-check}" in
        "check")
            perform_health_check
            ;;
        "status")
            show_detailed_status
            ;;
        "restart")
            restart_application
            ;;
        "logs")
            docker logs ${DOCKER_CONTAINER_NAME} 2>/dev/null || error "Could not retrieve container logs"
            ;;
        *)
            echo "Usage: $0 {check|status|restart|logs}"
            echo "  check   - Perform health check (default)"
            echo "  status  - Show detailed status"
            echo "  restart - Restart the application"
            echo "  logs    - Show container logs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 