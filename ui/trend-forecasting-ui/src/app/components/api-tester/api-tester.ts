import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { ApiTestService, ApiTestResult, ApiTestSuite } from '../../services/api-test.service';
import { ApiEndpoint, ApiRequest, ApiResponse } from '../../models/api.models';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { ValidationService } from '../../services/validation.service';
import { LoadingService } from '../../services/loading.service';
import { NotificationService } from '../../services/notification.service';

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
  // Backend integration testing
  integrationTestResults: ApiTestSuite | null = null;
  isRunningIntegrationTests = false;
  corsTestResult: ApiTestResult | null = null;
  proxyTestResults: ApiTestResult[] | null = null;

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
    private apiService: ApiService,
    private apiTestService: ApiTestService,
    private errorHandler: ErrorHandlerService,
    private validationService: ValidationService,
    private loadingService: LoadingService,
    private notificationService: NotificationService
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

    // Validate form before sending
    if (!this.validationService.validateForm(this.requestForm)) {
      const errors = this.validationService.getFormErrors(this.requestForm);
      this.errorHandler.handleValidationError(errors, 'Invalid Form Data');
      return;
    }

    // Validate JSON fields
    try {
      JSON.parse(this.requestForm.value.headers);
      if (this.requestForm.value.method !== 'GET') {
        JSON.parse(this.requestForm.value.body);
      }
    } catch (error) {
      this.errorHandler.handleGenericError(error, 'Invalid JSON Format');
      return;
    }

    await this.loadingService.withLoading(
      'api-request',
      async () => {
        this.error = null;
        this.response = null;

        const request: ApiRequest = {
          url: this.requestForm.value.baseUrl + this.requestForm.value.path,
          method: this.requestForm.value.method,
          headers: JSON.parse(this.requestForm.value.headers),
          body: this.requestForm.value.method !== 'GET' ? JSON.parse(this.requestForm.value.body) : undefined
        };

        const result = await this.apiService.sendRequest(request).toPromise();
        this.response = result || null;
        
        if (result && result.status >= 200 && result.status < 300) {
          this.notificationService.success('API request completed successfully');
        } else {
          this.notificationService.warning(`Request completed with status ${result?.status}`);
        }
      },
      {
        message: 'Sending API request...',
        showProgress: true,
        timeout: 30000
      }
    ).catch(error => {
      this.error = error.message || 'An error occurred';
      this.errorHandler.handleGenericError(error, 'API Request Error');
    });
  }

  formatJson(json: any): string {
    return JSON.stringify(json, null, 2);
  }

  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text);
  }

  // Backend Integration Testing Methods
  runIntegrationTests(): void {
    this.loadingService.withLoadingObservable(
      'integration-tests',
      this.apiTestService.testAllEndpoints(),
      {
        message: 'Running integration tests...',
        showProgress: true,
        timeout: 60000
      }
    ).subscribe({
      next: (results) => {
        this.integrationTestResults = results;
        this.notificationService.success('Integration tests completed');
      },
      error: (error) => {
        this.errorHandler.handleGenericError(error, 'Integration Test Error');
      }
    });
  }

  testCorsConfiguration(): void {
    this.loadingService.withLoadingObservable(
      'cors-test',
      this.apiTestService.testCorsConfiguration(),
      {
        message: 'Testing CORS configuration...',
        showProgress: false,
        timeout: 10000
      }
    ).subscribe({
      next: (result) => {
        this.corsTestResult = result;
        this.notificationService.success('CORS test completed');
      },
      error: (error) => {
        this.errorHandler.handleGenericError(error, 'CORS Test Error');
      }
    });
  }

  testProxyConfiguration(): void {
    this.loadingService.withLoadingObservable(
      'proxy-test',
      this.apiTestService.testProxyConfiguration(),
      {
        message: 'Testing proxy configuration...',
        showProgress: false,
        timeout: 10000
      }
    ).subscribe({
      next: (results) => {
        this.proxyTestResults = results;
        this.notificationService.success('Proxy test completed');
      },
      error: (error) => {
        this.errorHandler.handleGenericError(error, 'Proxy Test Error');
      }
    });
  }

  getStatusColor(status: string): string {
    switch (status) {
      case 'success': return 'success';
      case 'error': return 'warn';
      case 'timeout': return 'accent';
      default: return 'primary';
    }
  }

  getStatusIcon(status: string): string {
    switch (status) {
      case 'success': return 'check_circle';
      case 'error': return 'error';
      case 'timeout': return 'schedule';
      default: return 'help';
    }
  }
}
