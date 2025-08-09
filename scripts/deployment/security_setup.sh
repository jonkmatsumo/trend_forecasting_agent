#!/bin/bash

# Security Setup Script for Google Trends Quantile Forecaster
# This script configures security settings for EC2 deployment

set -e

# Configuration
APP_NAME="google-trends-forecaster"
APP_PORT=5000
DOMAIN_NAME=""  # Set this to your domain name
EMAIL=""        # Set this to your email for SSL certificates

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

# Function to configure firewall (iptables)
configure_firewall() {
    log "Configuring firewall rules..."
    
    # Flush existing rules
    sudo iptables -F
    sudo iptables -X
    
    # Set default policies
    sudo iptables -P INPUT DROP
    sudo iptables -P FORWARD DROP
    sudo iptables -P OUTPUT ACCEPT
    
    # Allow loopback
    sudo iptables -A INPUT -i lo -j ACCEPT
    
    # Allow established connections
    sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # Allow SSH (port 22)
    sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    
    # Allow HTTP (port 80) for SSL redirect
    sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    
    # Allow HTTPS (port 443)
    sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    
    # Allow application port (only from localhost for security)
    sudo iptables -A INPUT -p tcp --dport ${APP_PORT} -s 127.0.0.1 -j ACCEPT
    
    # Allow ICMP (ping)
    sudo iptables -A INPUT -p icmp -j ACCEPT
    
    # Save rules
    sudo service iptables save
    
    log "Firewall rules configured successfully"
}

# Function to install and configure SSL certificate with Let's Encrypt
setup_ssl_certificate() {
    if [ -z "$DOMAIN_NAME" ] || [ -z "$EMAIL" ]; then
        warning "Domain name or email not configured. Skipping SSL setup."
        return 0
    fi
    
    log "Setting up SSL certificate for domain: $DOMAIN_NAME"
    
    # Install certbot
    if ! command -v certbot >/dev/null 2>&1; then
        log "Installing certbot..."
        sudo yum install -y certbot python3-certbot-nginx
    fi
    
    # Create nginx configuration for the domain
    create_nginx_config
    
    # Obtain SSL certificate
    log "Obtaining SSL certificate..."
    sudo certbot --nginx -d $DOMAIN_NAME --email $EMAIL --agree-tos --non-interactive
    
    # Set up automatic renewal
    log "Setting up automatic certificate renewal..."
    (crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -
    
    log "SSL certificate setup completed"
}

# Function to create nginx configuration
create_nginx_config() {
    if [ -z "$DOMAIN_NAME" ]; then
        return 0
    fi
    
    log "Creating nginx configuration..."
    
    # Install nginx if not installed
    if ! command -v nginx >/dev/null 2>&1; then
        sudo yum install -y nginx
        sudo systemctl enable nginx
        sudo systemctl start nginx
    fi
    
    # Create nginx configuration file
    sudo tee /etc/nginx/conf.d/${DOMAIN_NAME}.conf > /dev/null <<EOF
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${DOMAIN_NAME};
    
    # SSL configuration will be added by certbot
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Proxy to application
    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:${APP_PORT}/health;
        access_log off;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    
    # Test nginx configuration
    sudo nginx -t
    
    # Reload nginx
    sudo systemctl reload nginx
    
    log "Nginx configuration created successfully"
}

# Function to configure system security
configure_system_security() {
    log "Configuring system security..."
    
    # Update system packages
    sudo yum update -y
    
    # Install security tools
    sudo yum install -y fail2ban
    
    # Configure fail2ban
    sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/secure
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3
EOF
    
    # Start and enable fail2ban
    sudo systemctl enable fail2ban
    sudo systemctl start fail2ban
    
    # Configure SSH security
    sudo tee -a /etc/ssh/sshd_config > /dev/null <<EOF

# Security settings
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOF
    
    # Restart SSH service
    sudo systemctl restart sshd
    
    log "System security configured successfully"
}

# Function to set up monitoring and logging
setup_monitoring() {
    log "Setting up monitoring and logging..."
    
    # Install monitoring tools
    sudo yum install -y htop iotop iftop
    
    # Create log rotation for application logs
    sudo tee /etc/logrotate.d/google-trends-forecaster > /dev/null <<EOF
/var/log/google-trends-forecaster/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ec2-user ec2-user
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
    endscript
}
EOF
    
    # Set up system monitoring cron job
    (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/google-trends-forecaster/scripts/deployment/health_check.sh check > /dev/null 2>&1") | crontab -
    
    log "Monitoring and logging setup completed"
}

# Function to create security documentation
create_security_docs() {
    log "Creating security documentation..."
    
    sudo tee /opt/google-trends-forecaster/SECURITY.md > /dev/null <<EOF
# Security Configuration

## Firewall Rules
- SSH (port 22): Allowed
- HTTP (port 80): Allowed (redirects to HTTPS)
- HTTPS (port 443): Allowed
- Application (port ${APP_PORT}): Only localhost access

## SSL Certificate
- Provider: Let's Encrypt
- Auto-renewal: Enabled (daily cron job)
- Domain: ${DOMAIN_NAME:-Not configured}

## Security Measures
- Fail2ban: Enabled for SSH and nginx
- Root login: Disabled
- Password authentication: Disabled
- SSH key authentication: Required
- Rate limiting: Enabled on API endpoints

## Monitoring
- Health checks: Every 5 minutes
- Log rotation: Daily
- Resource monitoring: Enabled

## Maintenance
- System updates: Manual (run: sudo yum update)
- Certificate renewal: Automatic
- Log cleanup: Automatic

## Emergency Contacts
- Email: ${EMAIL:-Not configured}
- Domain: ${DOMAIN_NAME:-Not configured}

Last updated: $(date)
EOF
    
    log "Security documentation created"
}

# Function to show security status
show_security_status() {
    log "=== Security Status ==="
    
    # Firewall status
    log "Firewall Rules:"
    sudo iptables -L -n | head -20
    
    # SSL certificate status
    if [ -n "$DOMAIN_NAME" ]; then
        log "SSL Certificate Status:"
        sudo certbot certificates 2>/dev/null || warning "No SSL certificates found"
    fi
    
    # Fail2ban status
    log "Fail2ban Status:"
    sudo systemctl status fail2ban --no-pager -l
    
    # SSH configuration
    log "SSH Configuration:"
    grep -E "^(PermitRootLogin|PasswordAuthentication|PubkeyAuthentication)" /etc/ssh/sshd_config
    
    # Nginx status
    log "Nginx Status:"
    sudo systemctl status nginx --no-pager -l
}

# Main function
main() {
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
    
    # Parse command line arguments
    case "${1:-setup}" in
        "setup")
            log "Starting security setup..."
            configure_firewall
            configure_system_security
            setup_ssl_certificate
            setup_monitoring
            create_security_docs
            log "Security setup completed successfully!"
            ;;
        "status")
            show_security_status
            ;;
        "ssl")
            setup_ssl_certificate
            ;;
        "firewall")
            configure_firewall
            ;;
        *)
            echo "Usage: $0 {setup|status|ssl|firewall}"
            echo "  setup    - Complete security setup (default)"
            echo "  status   - Show security status"
            echo "  ssl      - Setup SSL certificate only"
            echo "  firewall - Configure firewall only"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 