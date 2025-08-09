#!/bin/bash

# EC2 Instance Setup Script for Google Trends Quantile Forecaster
# This script sets up an EC2 instance with Docker and necessary dependencies

set -e  # Exit on any error

echo "Starting EC2 instance setup..."

# Update system packages
echo "Updating system packages..."
sudo yum update -y

# Install Docker
echo "Installing Docker..."
sudo yum install -y docker

# Start and enable Docker service
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
echo "Adding user to docker group..."
sudo usermod -a -G docker ec2-user

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/google-trends-forecaster
sudo chown ec2-user:ec2-user /opt/google-trends-forecaster

# Install additional utilities
echo "Installing additional utilities..."
sudo yum install -y git curl wget

# Create log directory
echo "Creating log directory..."
sudo mkdir -p /var/log/google-trends-forecaster
sudo chown ec2-user:ec2-user /var/log/google-trends-forecaster

# Set up log rotation
echo "Setting up log rotation..."
sudo tee /etc/logrotate.d/google-trends-forecaster > /dev/null <<EOF
/var/log/google-trends-forecaster/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ec2-user ec2-user
}
EOF

echo "EC2 instance setup completed successfully!"
echo "Please log out and log back in for Docker group changes to take effect." 