import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { ApiEndpoint, ApiRequest, ApiResponse } from '../../models/api.models';

@Component({
  selector: 'app-api-tester',
  templateUrl: './api-tester.html',
  styleUrls: ['./api-tester.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule
  ],
  standalone: true
})
export class ApiTesterComponent implements OnInit {
  endpoints: ApiEndpoint[] = [
    {
      name: 'Health Check',
      method: 'GET',
      path: '/health',
      description: 'API health check endpoint'
    },
    {
      name: 'Get Trends',
      method: 'POST',
      path: '/trends',
      description: 'Get Google Trends data for keywords',
      bodyTemplate: {
        keywords: ['machine learning', 'artificial intelligence'],
        geo: 'US',
        timeframe: 'today 12-m',
        category: 0
      }
    },
    {
      name: 'Train Model',
      method: 'POST',
      path: '/models/train',
      description: 'Train a forecasting model',
      bodyTemplate: {
        keywords: ['python'],
        geo: 'US',
        timeframe: 'today 12-m',
        model_type: 'prophet',
        forecast_horizon: 30
      }
    },
    {
      name: 'Model Prediction',
      method: 'POST',
      path: '/models/{model_id}/predict',
      description: 'Make predictions with a trained model',
      bodyTemplate: {
        horizon: 30,
        quantiles: [0.1, 0.5, 0.9]
      }
    },
    {
      name: 'Get Model',
      method: 'GET',
      path: '/models/{model_id}',
      description: 'Get model details and status'
    },
    {
      name: 'List Models',
      method: 'GET',
      path: '/models',
      description: 'List all trained models'
    },
    {
      name: 'Clear Cache',
      method: 'POST',
      path: '/trends/cache/clear',
      description: 'Clear trends data cache'
    },
    {
      name: 'Cache Stats',
      method: 'GET',
      path: '/trends/cache/stats',
      description: 'Get cache statistics'
    },
    {
      name: 'Trends Summary',
      method: 'POST',
      path: '/trends/summary',
      description: 'Get trends summary data',
      bodyTemplate: {
        keywords: ['python'],
        geo: 'US',
        timeframe: 'today 12-m'
      }
    },
    {
      name: 'Compare Trends',
      method: 'POST',
      path: '/trends/compare',
      description: 'Compare multiple keywords',
      bodyTemplate: {
        keywords: ['python', 'javascript'],
        geo: 'US',
        timeframe: 'today 12-m'
      }
    }
  ];

  selectedEndpoint: ApiEndpoint | null = null;
  requestForm: FormGroup;
  response: ApiResponse | null = null;
  isLoading = false;
  error: string | null = null;

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService
  ) {
    this.requestForm = this.fb.group({
      baseUrl: ['http://localhost:5000', Validators.required],
      path: ['', Validators.required],
      method: ['GET', Validators.required],
      headers: ['{"Content-Type": "application/json"}'],
      body: ['{}']
    });
  }

  ngOnInit(): void {
    this.selectEndpoint(this.endpoints[0]);
  }

  selectEndpoint(endpoint: ApiEndpoint): void {
    this.selectedEndpoint = endpoint;
    this.requestForm.patchValue({
      path: endpoint.path,
      method: endpoint.method,
      body: endpoint.bodyTemplate ? JSON.stringify(endpoint.bodyTemplate, null, 2) : '{}'
    });
  }

  async sendRequest(): Promise<void> {
    if (this.requestForm.invalid || this.isLoading) return;

    this.isLoading = true;
    this.error = null;
    this.response = null;

    try {
      const request: ApiRequest = {
        url: this.requestForm.value.baseUrl + this.requestForm.value.path,
        method: this.requestForm.value.method,
        headers: JSON.parse(this.requestForm.value.headers),
        body: this.requestForm.value.method !== 'GET' ? JSON.parse(this.requestForm.value.body) : undefined
      };

      const result = await this.apiService.sendRequest(request).toPromise();
      this.response = result || null;
    } catch (error: any) {
      this.error = error.message || 'An error occurred';
      console.error('API request error:', error);
    } finally {
      this.isLoading = false;
    }
  }

  formatJson(json: any): string {
    return JSON.stringify(json, null, 2);
  }

  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text);
  }
}
