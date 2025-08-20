import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiTestService, ApiTestSuite, ApiTestResult } from './api-test.service';
import { environment } from '../../environments/environment';

describe('ApiTestService', () => {
  let service: ApiTestService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        ApiTestService
      ]
    });
    service = TestBed.inject(ApiTestService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    if (httpMock) {
      httpMock.verify();
    }
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('testAllEndpoints', () => {
    it('should test all endpoints and return test suite', () => {
      const mockResponses = {
        '/health': { status: 'ok' },
        '/api/health': { status: 'ok' },
        '/agent/health': { status: 'ok' },
        '/agent/ask': { response: 'test' },
        '/api/trends': { trends: [] }
      };

      service.testAllEndpoints().subscribe((suite: ApiTestSuite) => {
        expect(suite.name).toBe('Full API Test Suite');
        expect(suite.description).toContain('Comprehensive API test suite');
        expect(suite.tests.length).toBe(6);
        expect(suite.summary.total).toBe(6);
        expect(suite.summary.successful).toBe(6);
        expect(suite.summary.failed).toBe(0);
        expect(suite.summary.averageResponseTime).toBeGreaterThan(0);
      });

      // Health endpoints
      const healthReq = httpMock.expectOne(`${environment.apiUrl}/health`);
      healthReq.flush(mockResponses['/health']);

      const apiHealthReq = httpMock.expectOne(`${environment.apiUrl}/api/health`);
      apiHealthReq.flush(mockResponses['/api/health']);

      const agentHealthReq = httpMock.expectOne(`${environment.apiUrl}/agent/health`);
      agentHealthReq.flush(mockResponses['/agent/health']);

      // Agent endpoint
      const agentAskReq = httpMock.expectOne(`${environment.agentUrl}/ask`);
      agentAskReq.flush(mockResponses['/agent/ask']);

      // API endpoints
      const trendsGetReq = httpMock.expectOne(`${environment.apiUrl}/api/trends`);
      trendsGetReq.flush(mockResponses['/api/trends']);

      const trendsPostReq = httpMock.expectOne(`${environment.apiUrl}/api/trends`);
      trendsPostReq.flush(mockResponses['/api/trends']);
    });

    it('should handle mixed success and failure results', () => {
      service.testAllEndpoints().subscribe((suite: ApiTestSuite) => {
        expect(suite.summary.total).toBe(6);
        expect(suite.summary.successful).toBe(1);
        expect(suite.summary.failed).toBe(5);
      });

      // Health endpoint - success
      const healthReq = httpMock.expectOne(`${environment.apiUrl}/health`);
      healthReq.flush({ status: 'ok' });

      // Other endpoints - errors
      const apiHealthReq = httpMock.expectOne(`${environment.apiUrl}/api/health`);
      apiHealthReq.error(new ErrorEvent('Network error'), { status: 500 });

      const agentHealthReq = httpMock.expectOne(`${environment.apiUrl}/agent/health`);
      agentHealthReq.error(new ErrorEvent('Network error'), { status: 500 });

      const agentAskReq = httpMock.expectOne(`${environment.agentUrl}/ask`);
      agentAskReq.error(new ErrorEvent('Network error'), { status: 500 });

      const trendsGetReq = httpMock.expectOne(`${environment.apiUrl}/api/trends`);
      trendsGetReq.error(new ErrorEvent('Network error'), { status: 500 });

      const trendsPostReq = httpMock.expectOne(`${environment.apiUrl}/api/trends`);
      trendsPostReq.error(new ErrorEvent('Network error'), { status: 500 });
    });
  });

  describe('testCorsConfiguration', () => {
    it('should test CORS configuration successfully', () => {
      const mockResponse = { status: 'ok' };

      service.testCorsConfiguration().subscribe((result: ApiTestResult) => {
        expect(result.endpoint).toBe('CORS Test');
        expect(result.method).toBe('GET');
        expect(result.status).toBe('success');
        expect(result.statusCode).toBe(200);
        expect(result.responseTime).toBeGreaterThan(0);
        expect(result.data).toBeDefined();
        expect(result.data.corsHeaders).toBeDefined();
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.flush(mockResponse, {
        headers: {
          'access-control-allow-origin': 'http://localhost:4200',
          'access-control-allow-methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'access-control-allow-headers': 'Content-Type, Authorization'
        }
      });
    });

    it('should handle CORS test errors', () => {
      service.testCorsConfiguration().subscribe((result: ApiTestResult) => {
        expect(result.endpoint).toBe('CORS Test');
        expect(result.method).toBe('GET');
        expect(result.status).toBe('error');
        expect(result.responseTime).toBeGreaterThan(0);
        expect(result.error).toBeDefined();
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.error(new ErrorEvent('Network error'), { status: 500 });
    });
  });

  describe('testProxyConfiguration', () => {
    it('should test proxy configuration successfully', () => {
      const mockResponses = {
        '/health': { status: 'ok' },
        '/api/health': { status: 'ok' },
        '/agent/health': { status: 'ok' }
      };

      service.testProxyConfiguration().subscribe((results: ApiTestResult[]) => {
        expect(results.length).toBe(3);
        expect(results.every(r => r.status === 'success')).toBe(true);
        expect(results.every(r => r.responseTime && r.responseTime > 0)).toBe(true);
      });

      // Test API proxy
      const healthReq = httpMock.expectOne(`${environment.apiUrl}/health`);
      healthReq.flush(mockResponses['/health']);

      const apiHealthReq = httpMock.expectOne(`${environment.apiUrl}/api/health`);
      apiHealthReq.flush(mockResponses['/api/health']);

      const agentHealthReq = httpMock.expectOne(`${environment.apiUrl}/agent/health`);
      agentHealthReq.flush(mockResponses['/agent/health']);
    });

    it('should handle proxy configuration issues', () => {
      service.testProxyConfiguration().subscribe((results: ApiTestResult[]) => {
        expect(results.length).toBe(3);
        expect(results.some(r => r.status === 'error')).toBe(true);
        expect(results.some(r => r.status === 'success')).toBe(true);
      });

      // API proxy fails
      const healthReq = httpMock.expectOne(`${environment.apiUrl}/health`);
      healthReq.error(new ErrorEvent('Network error'), { status: 500 });

      // Other proxies succeed
      const apiHealthReq = httpMock.expectOne(`${environment.apiUrl}/api/health`);
      apiHealthReq.flush({ status: 'ok' });

      const agentHealthReq = httpMock.expectOne(`${environment.apiUrl}/agent/health`);
      agentHealthReq.flush({ status: 'ok' });
    });
  });
}); 