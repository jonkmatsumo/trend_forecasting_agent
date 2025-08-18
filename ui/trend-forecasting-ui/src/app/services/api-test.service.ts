import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { catchError, map, timeout } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface ApiTestResult {
  endpoint: string;
  method: string;
  status: 'success' | 'error' | 'timeout';
  statusCode?: number;
  responseTime?: number;
  data?: any;
  error?: string;
}

export interface ApiTestSuite {
  name: string;
  description: string;
  tests: ApiTestResult[];
  summary: {
    total: number;
    successful: number;
    failed: number;
    averageResponseTime: number;
  };
}

@Injectable({
  providedIn: 'root'
})
export class ApiTestService {
  private readonly TIMEOUT_MS = 10000; // 10 seconds

  constructor(private http: HttpClient) {}

  /**
   * Test all available API endpoints
   */
  testAllEndpoints(): Observable<ApiTestSuite> {
    const tests: Observable<ApiTestResult>[] = [];

    // Health endpoints
    tests.push(this.testEndpoint('GET', '/health', 'Health Check'));
    tests.push(this.testEndpoint('GET', '/api/health', 'API Health Check'));
    tests.push(this.testEndpoint('GET', '/agent/health', 'Agent Health Check'));

    // Agent endpoints
    tests.push(this.testAgentEndpoint('POST', '/agent/ask', 'Agent Ask', {
      query: 'What is the health status?'
    }));

    // API endpoints
    tests.push(this.testEndpoint('GET', '/api/trends', 'Get Trends (GET)'));
    tests.push(this.testEndpoint('POST', '/api/trends', 'Get Trends (POST)', {
      keywords: ['python', 'javascript'],
      timeframe: 'today 12-m',
      geo: ''
    }));

    return new Observable(observer => {
      const results: ApiTestResult[] = [];
      let completed = 0;

      tests.forEach(test => {
        test.subscribe({
          next: (result) => {
            results.push(result);
            completed++;
            if (completed === tests.length) {
              const suite = this.createTestSuite('Full API Test Suite', results);
              observer.next(suite);
              observer.complete();
            }
          },
          error: (error) => {
            results.push({
              endpoint: 'Test Error',
              method: 'GET',
              status: 'error',
              error: error.message
            });
            completed++;
            if (completed === tests.length) {
              const suite = this.createTestSuite('Full API Test Suite', results);
              observer.next(suite);
              observer.complete();
            }
          }
        });
      });
    });
  }

  /**
   * Test a specific endpoint
   */
  private testEndpoint(method: string, endpoint: string, description: string, body?: any): Observable<ApiTestResult> {
    const startTime = Date.now();
    const url = `${environment.apiUrl}${endpoint}`;

    let request: Observable<any>;

    switch (method) {
      case 'GET':
        request = this.http.get(url);
        break;
      case 'POST':
        request = this.http.post(url, body);
        break;
      case 'PUT':
        request = this.http.put(url, body);
        break;
      case 'DELETE':
        request = this.http.delete(url);
        break;
      default:
        return of({
          endpoint: `${method} ${endpoint}`,
          method,
          status: 'error' as const,
          error: `Unsupported method: ${method}`
        });
    }

    return request.pipe(
      timeout(this.TIMEOUT_MS),
      map(response => ({
        endpoint: `${method} ${endpoint}`,
        method,
        status: 'success' as const,
        statusCode: 200,
        responseTime: Date.now() - startTime,
        data: response
      })),
      catchError(error => {
        const result: ApiTestResult = {
          endpoint: `${method} ${endpoint}`,
          method,
          status: error.name === 'TimeoutError' ? 'timeout' : 'error',
          responseTime: Date.now() - startTime,
          error: error.message || 'Unknown error'
        };

        if (error.status) {
          result.statusCode = error.status;
        }

        return of(result);
      })
    );
  }

  /**
   * Test agent-specific endpoints
   */
  private testAgentEndpoint(method: string, endpoint: string, description: string, body?: any): Observable<ApiTestResult> {
    const startTime = Date.now();
    const url = `${environment.agentUrl}${endpoint}`;

    let request: Observable<any>;

    switch (method) {
      case 'GET':
        request = this.http.get(url);
        break;
      case 'POST':
        request = this.http.post(url, body);
        break;
      default:
        return of({
          endpoint: `${method} ${endpoint}`,
          method,
          status: 'error' as const,
          error: `Unsupported method: ${method}`
        });
    }

    return request.pipe(
      timeout(this.TIMEOUT_MS),
      map(response => ({
        endpoint: `${method} ${endpoint}`,
        method,
        status: 'success' as const,
        statusCode: 200,
        responseTime: Date.now() - startTime,
        data: response
      })),
      catchError(error => {
        const result: ApiTestResult = {
          endpoint: `${method} ${endpoint}`,
          method,
          status: error.name === 'TimeoutError' ? 'timeout' : 'error',
          responseTime: Date.now() - startTime,
          error: error.message || 'Unknown error'
        };

        if (error.status) {
          result.statusCode = error.status;
        }

        return of(result);
      })
    );
  }

  /**
   * Create a test suite summary
   */
  private createTestSuite(name: string, tests: ApiTestResult[]): ApiTestSuite {
    const successful = tests.filter(t => t.status === 'success').length;
    const failed = tests.filter(t => t.status !== 'success').length;
    const responseTimes = tests
      .filter(t => t.responseTime !== undefined)
      .map(t => t.responseTime!);

    const averageResponseTime = responseTimes.length > 0
      ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length
      : 0;

    return {
      name,
      description: `Comprehensive API test suite with ${tests.length} endpoints`,
      tests,
      summary: {
        total: tests.length,
        successful,
        failed,
        averageResponseTime: Math.round(averageResponseTime)
      }
    };
  }

  /**
   * Test CORS configuration
   */
  testCorsConfiguration(): Observable<ApiTestResult> {
    const startTime = Date.now();
    const url = `${environment.apiUrl}/health`;

    return this.http.get(url, { observe: 'response' }).pipe(
      timeout(this.TIMEOUT_MS),
      map(response => ({
        endpoint: 'CORS Test',
        method: 'GET',
        status: 'success' as const,
        statusCode: response.status,
        responseTime: Date.now() - startTime,
        data: {
          corsHeaders: {
            'access-control-allow-origin': response.headers.get('access-control-allow-origin'),
            'access-control-allow-methods': response.headers.get('access-control-allow-methods'),
            'access-control-allow-headers': response.headers.get('access-control-allow-headers')
          }
        }
      })),
      catchError(error => {
        return of({
          endpoint: 'CORS Test',
          method: 'GET',
          status: 'error' as const,
          responseTime: Date.now() - startTime,
          error: error.message || 'CORS test failed'
        });
      })
    );
  }

  /**
   * Test proxy configuration
   */
  testProxyConfiguration(): Observable<ApiTestResult[]> {
    const tests: Observable<ApiTestResult>[] = [];

    // Test direct API calls (should work with proxy)
    tests.push(this.testEndpoint('GET', '/health', 'Proxy Health Check'));
    tests.push(this.testEndpoint('GET', '/api/health', 'Proxy API Health Check'));
    tests.push(this.testEndpoint('GET', '/agent/health', 'Proxy Agent Health Check'));

    return new Observable(observer => {
      const results: ApiTestResult[] = [];
      let completed = 0;

      tests.forEach(test => {
        test.subscribe({
          next: (result) => {
            results.push(result);
            completed++;
            if (completed === tests.length) {
              observer.next(results);
              observer.complete();
            }
          },
          error: (error) => {
            results.push({
              endpoint: 'Proxy Test',
              method: 'GET',
              status: 'error',
              error: error.message
            });
            completed++;
            if (completed === tests.length) {
              observer.next(results);
              observer.complete();
            }
          }
        });
      });
    });
  }
} 