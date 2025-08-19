import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, catchError } from 'rxjs';
import { ApiRequest, ApiResponse } from '../models/api.models';
import { ErrorHandlerService } from './error-handler.service';
import { ValidationService } from './validation.service';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService,
    private validationService: ValidationService
  ) {}

  sendRequest(request: ApiRequest): Observable<ApiResponse> {
    const startTime = Date.now();
    
    // Validate request before sending
    const validationErrors = this.validationService.validateApiRequest(request);
    if (validationErrors.length > 0) {
      this.errorHandler.handleValidationError(validationErrors, 'Invalid API Request');
      return new Observable(observer => {
        const apiResponse: ApiResponse = {
          status: 400,
          statusText: 'Bad Request',
          data: { errors: validationErrors },
          headers: {},
          responseTime: Date.now() - startTime
        };
        observer.next(apiResponse);
        observer.complete();
      });
    }
    
    const options = {
      headers: new HttpHeaders(request.headers),
      observe: 'response' as const
    };

    let httpCall: Observable<any>;
    
    switch (request.method) {
      case 'GET':
        httpCall = this.http.get(request.url, options);
        break;
      case 'POST':
        httpCall = this.http.post(request.url, request.body, options);
        break;
      case 'PUT':
        httpCall = this.http.put(request.url, request.body, options);
        break;
      case 'DELETE':
        httpCall = this.http.delete(request.url, options);
        break;
      default:
        throw new Error(`Unsupported HTTP method: ${request.method}`);
    }

    return this.errorHandler.createRetryableRequest<ApiResponse>(
      new Observable(observer => {
        httpCall.subscribe({
          next: (response) => {
            const apiResponse: ApiResponse = {
              status: response.status,
              statusText: response.statusText,
              data: response.body,
              headers: response.headers,
              responseTime: Date.now() - startTime
            };
            observer.next(apiResponse);
            observer.complete();
          },
          error: (error: HttpErrorResponse) => {
            const apiResponse: ApiResponse = {
              status: error.status || 0,
              statusText: error.statusText || 'Unknown Error',
              data: error.error || error.message,
              headers: error.headers,
              responseTime: Date.now() - startTime
            };
            observer.next(apiResponse);
            observer.complete();
          }
        });
      }),
      { retryAttempts: 2, customMessage: 'API request failed' }
    );
  }
} 