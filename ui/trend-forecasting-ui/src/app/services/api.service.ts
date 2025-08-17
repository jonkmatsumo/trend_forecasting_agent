import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiRequest, ApiResponse } from '../models/api.models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(private http: HttpClient) {}

  sendRequest(request: ApiRequest): Observable<ApiResponse> {
    const startTime = Date.now();
    
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

    return new Observable(observer => {
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
        error: (error) => {
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
    });
  }
} 