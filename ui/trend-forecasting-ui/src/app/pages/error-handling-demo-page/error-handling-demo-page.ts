import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatDividerModule } from '@angular/material/divider';
import { CommonModule } from '@angular/common';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { ValidationService } from '../../services/validation.service';
import { LoadingService } from '../../services/loading.service';
import { NotificationService } from '../../services/notification.service';
import { SkeletonLoaderComponent } from '../../components/shared/skeleton-loader/skeleton-loader';

@Component({
  selector: 'app-error-handling-demo-page',
  templateUrl: './error-handling-demo-page.html',
  styleUrls: ['./error-handling-demo-page.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatCardModule,
    MatTabsModule,
    MatDividerModule,
    SkeletonLoaderComponent
  ],
  standalone: true
})
export class ErrorHandlingDemoPageComponent implements OnInit {
  // Form validation demo
  validationForm: FormGroup;
  validationErrors: any[] = [];

  // Error handling demo
  errorDemoResults: any[] = [];

  // Loading states demo
  loadingDemoStates: { [key: string]: boolean } = {};
  skeletonDemoData: any[] = [];

  // Demo data
  demoEndpoints = [
    { name: 'Health Check', url: '/health', method: 'GET' },
    { name: 'Invalid Endpoint', url: '/invalid', method: 'GET' },
    { name: 'Timeout Test', url: '/timeout', method: 'GET' },
    { name: 'Server Error', url: '/error', method: 'GET' }
  ];

  constructor(
    private fb: FormBuilder,
    private errorHandler: ErrorHandlerService,
    private validationService: ValidationService,
    private loadingService: LoadingService,
    private notificationService: NotificationService
  ) {
    this.validationForm = this.fb.group({
      name: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      url: ['', [ValidationService.validators.url()]],
      json: ['', [ValidationService.validators.json()]],
      number: ['', [ValidationService.validators.range(1, 100)]],
      keywords: ['', [ValidationService.validators.keywordsArray()]],
      timeframe: ['', [ValidationService.validators.timeframe()]],
      modelType: ['', [ValidationService.validators.modelType()]]
    });
  }

  ngOnInit(): void {
    this.loadSkeletonDemoData();
  }

  // Form Validation Demo
  testFormValidation(): void {
    this.validationErrors = [];
    
    if (this.validationForm.valid) {
      this.notificationService.success('Form validation passed!');
    } else {
      const errors = this.validationService.getFormErrors(this.validationForm);
      this.errorHandler.handleValidationError(errors, 'Form Validation Failed');
      this.validationErrors = errors;
    }
  }

  resetForm(): void {
    this.validationForm.reset();
    this.validationErrors = [];
  }

  // Error Handling Demo
  testHttpError(status: number): void {
    const error = {
      status,
      statusText: this.getStatusText(status),
      message: this.getErrorMessage(status)
    };

    this.errorDemoResults.push({
      timestamp: new Date(),
      error,
      type: 'HTTP Error'
    });

    this.errorHandler.handleHttpError(error as any, {
      customMessage: `Simulated ${status} error`,
      showNotification: true
    });
  }

  testValidationError(): void {
    const errors = [
      { field: 'email', message: 'Invalid email format', code: 'email' },
      { field: 'password', message: 'Password must be at least 8 characters', code: 'minlength' },
      { field: 'url', message: 'Invalid URL format', code: 'invalidUrl' }
    ];

    this.errorDemoResults.push({
      timestamp: new Date(),
      errors,
      type: 'Validation Error'
    });

    this.errorHandler.handleValidationError(errors, 'Validation Error Demo');
  }

  testGenericError(): void {
    const error = new Error('This is a simulated generic error for demonstration purposes');
    
    this.errorDemoResults.push({
      timestamp: new Date(),
      error: error.message,
      type: 'Generic Error'
    });

    this.errorHandler.handleGenericError(error, 'Generic Error Demo');
  }

  clearErrorResults(): void {
    this.errorDemoResults = [];
  }

  // Loading States Demo
  testLoadingState(id: string, duration: number = 3000): void {
    this.loadingService.startLoading(id, {
      message: `Loading ${id}...`,
      showProgress: true,
      timeout: duration + 1000
    });

    // Simulate progress updates
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      this.loadingService.updateProgress(id, progress);
      
      if (progress >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          this.loadingService.stopLoading(id, 'Completed successfully');
        }, 500);
      }
    }, duration / 10);
  }

  testSkeletonLoading(): void {
    this.loadingService.showSkeleton('skeleton-demo', 'Loading content...');
    
    setTimeout(() => {
      this.loadingService.hideSkeleton('skeleton-demo');
    }, 3000);
  }

  // Helper methods
  private getStatusText(status: number): string {
    const statusTexts: { [key: number]: string } = {
      400: 'Bad Request',
      401: 'Unauthorized',
      403: 'Forbidden',
      404: 'Not Found',
      408: 'Request Timeout',
      429: 'Too Many Requests',
      500: 'Internal Server Error',
      502: 'Bad Gateway',
      503: 'Service Unavailable',
      504: 'Gateway Timeout'
    };
    return statusTexts[status] || 'Unknown Error';
  }

  private getErrorMessage(status: number): string {
    const messages: { [key: number]: string } = {
      400: 'The request was malformed or invalid',
      401: 'Authentication is required to access this resource',
      403: 'You do not have permission to access this resource',
      404: 'The requested resource was not found',
      408: 'The request timed out',
      429: 'Too many requests, please try again later',
      500: 'An internal server error occurred',
      502: 'The server is temporarily unavailable',
      503: 'The service is temporarily down for maintenance',
      504: 'The server took too long to respond'
    };
    return messages[status] || 'An unexpected error occurred';
  }

  private loadSkeletonDemoData(): void {
    // Simulate loading data for skeleton demo
    this.skeletonDemoData = [
      { id: 1, name: 'Loading...', description: 'Loading...', status: 'Loading...' },
      { id: 2, name: 'Loading...', description: 'Loading...', status: 'Loading...' },
      { id: 3, name: 'Loading...', description: 'Loading...', status: 'Loading...' }
    ];

    setTimeout(() => {
      this.skeletonDemoData = [
        { id: 1, name: 'Machine Learning', description: 'AI and ML trends', status: 'Active' },
        { id: 2, name: 'Python Programming', description: 'Python development trends', status: 'Active' },
        { id: 3, name: 'Data Science', description: 'Data science and analytics', status: 'Inactive' }
      ];
    }, 3000);
  }

  // Notification demo
  showSuccessNotification(): void {
    this.notificationService.success('This is a success notification!', 'Success');
  }

  showErrorNotification(): void {
    this.notificationService.error('This is an error notification!', 'Error');
  }

  showWarningNotification(): void {
    this.notificationService.warning('This is a warning notification!', 'Warning');
  }

  showInfoNotification(): void {
    this.notificationService.info('This is an info notification!', 'Information');
  }
} 