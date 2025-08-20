import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiService } from './api.service';
import { ErrorHandlerService } from './error-handler.service';
import { ValidationService } from './validation.service';
import { ApiRequest, ApiResponse } from '../models/api.models';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;
  let errorHandler: jasmine.SpyObj<ErrorHandlerService>;
  let validationService: jasmine.SpyObj<ValidationService>;
  const baseUrl = 'http://localhost:5000';

  beforeEach(() => {
    const errorHandlerSpy = jasmine.createSpyObj('ErrorHandlerService', ['handleValidationError', 'createRetryableRequest']);
    const validationServiceSpy = jasmine.createSpyObj('ValidationService', ['validateApiRequest']);

    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        ApiService,
        { provide: ErrorHandlerService, useValue: errorHandlerSpy },
        { provide: ValidationService, useValue: validationServiceSpy }
      ]
    });
    
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
    errorHandler = TestBed.inject(ErrorHandlerService) as jasmine.SpyObj<ErrorHandlerService>;
    validationService = TestBed.inject(ValidationService) as jasmine.SpyObj<ValidationService>;
    
    // Setup default spy behavior
    validationService.validateApiRequest.and.returnValue([]);
    errorHandler.createRetryableRequest.and.callFake((observable: any) => observable);
  });

  afterEach(() => {
    if (httpMock) {
      httpMock.verify();
    }
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('sendRequest', () => {
    const mockRequest: ApiRequest = {
      method: 'GET',
      url: `${baseUrl}/api/test`,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const mockResponse: ApiResponse = {
      status: 200,
      statusText: 'OK',
      headers: {
        'content-type': 'application/json'
      },
      data: {
        message: 'Success',
        data: { test: 'value' }
      },
      responseTime: 150
    };

    it('should send GET request to correct endpoint', () => {
      service.sendRequest(mockRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      expect(req.request.method).toBe('GET');
      expect(req.request.headers.get('Content-Type')).toBe('application/json');
    });

    it('should return API response on success', () => {
      service.sendRequest(mockRequest).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(200);
        expect(response.data).toEqual(mockResponse.data);
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      req.flush(mockResponse.data, {
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'application/json' }
      });
    });

    it('should handle POST requests with body', () => {
      const postRequest: ApiRequest = {
        ...mockRequest,
        method: 'POST',
        body: { test: 'data' }
      };

      service.sendRequest(postRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ test: 'data' });
    });

    it('should handle PUT requests', () => {
      const putRequest: ApiRequest = {
        ...mockRequest,
        method: 'PUT',
        body: { update: 'data' }
      };

      service.sendRequest(putRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      expect(req.request.method).toBe('PUT');
      expect(req.request.body).toEqual({ update: 'data' });
    });

    it('should handle DELETE requests', () => {
      const deleteRequest: ApiRequest = {
        ...mockRequest,
        method: 'DELETE'
      };

      service.sendRequest(deleteRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      expect(req.request.method).toBe('DELETE');
    });

    it('should handle HTTP errors', () => {
      service.sendRequest(mockRequest).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(404);
        expect(response.statusText).toBe('Not Found');
      });

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      req.flush('Not found', { status: 404, statusText: 'Not Found' });
    });

    it('should handle network errors', () => {
      service.sendRequest(mockRequest).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(0);
        expect(response.statusText).toBe('Unknown Error');
      });

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      req.error(new ErrorEvent('Network error'));
    });

    it('should handle timeout errors', () => {
      service.sendRequest(mockRequest).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(408);
        expect(response.statusText).toBe('Request Timeout');
      });

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      req.flush('Timeout', { status: 408, statusText: 'Request Timeout' });
    });

    it('should handle server errors', () => {
      service.sendRequest(mockRequest).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(500);
        expect(response.statusText).toBe('Internal Server Error');
      });

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      req.flush('Server error', { status: 500, statusText: 'Internal Server Error' });
    });

    it('should throw error for unsupported HTTP method', () => {
      const invalidRequest: ApiRequest = {
        method: 'PATCH' as any,
        url: `${baseUrl}/api/test`,
        headers: {}
      };

      expect(() => {
        service.sendRequest(invalidRequest).subscribe();
      }).toThrow('Unsupported HTTP method: PATCH');
    });

    it('should include custom headers in request', () => {
      const requestWithHeaders: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/test`,
        headers: { 'Authorization': 'Bearer token123' }
      };

      service.sendRequest(requestWithHeaders).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/test`);
      expect(req.request.headers.get('Authorization')).toBe('Bearer token123');
      req.flush({ message: 'Success' });
    });
  });

  describe('Request Validation', () => {
    it('should handle requests without headers', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/simple`,
        headers: {}
      };

      service.sendRequest(request).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/simple`);
      expect(req.request.method).toBe('GET');
      req.flush({ message: 'Success' });
    });

    it('should handle requests with empty headers', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/simple`,
        headers: {}
      };

      service.sendRequest(request).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/simple`);
      expect(req.request.method).toBe('GET');
      req.flush({ message: 'Success' });
    });
  });

  describe('Response Processing', () => {
    it('should handle JSON responses', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/json`,
        headers: {}
      };

      const jsonResponse = {
        data: { users: [{ id: 1, name: 'John' }] },
        meta: { total: 1 }
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.data).toEqual(jsonResponse);
        expect(response.status).toBe(200);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/json`);
      req.flush(jsonResponse, {
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'application/json' }
      });
    });

    it('should handle text responses', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/text`,
        headers: {}
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.data).toBe('Plain text response');
        expect(response.status).toBe(200);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/text`);
      req.flush('Plain text response', {
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'text/plain' }
      });
    });

    it('should handle empty responses', () => {
      const request: ApiRequest = {
        method: 'DELETE',
        url: `${baseUrl}/api/empty`,
        headers: {}
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.data).toBeNull();
        expect(response.status).toBe(204);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/empty`);
      req.flush(null, {
        status: 204,
        statusText: 'No Content'
      });
    });
  });

  describe('Performance', () => {
    it('should handle multiple concurrent requests', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/concurrent`,
        headers: {}
      };

      const responses: ApiResponse[] = [];

      // Send multiple concurrent requests
      for (let i = 0; i < 5; i++) {
        service.sendRequest(request).subscribe((response: ApiResponse) => {
          responses.push(response);
        });
      }

      // Verify all requests were made
      const requests = httpMock.match(`${baseUrl}/api/concurrent`);
      expect(requests.length).toBe(5);

      // Respond to all requests
      requests.forEach((req, index) => {
        req.flush({
          message: `Response ${index + 1}`,
          data: { index }
        }, {
          status: 200,
          statusText: 'OK',
          headers: { 'content-type': 'application/json' }
        });
      });

      expect(responses.length).toBe(5);
    });

    it('should measure response time accurately', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${baseUrl}/api/timing`,
        headers: {}
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/timing`);
      
      // Simulate some delay
      setTimeout(() => {
        req.flush({ message: 'Success' }, {
          status: 200,
          statusText: 'OK',
          headers: { 'content-type': 'application/json' }
        });
      }, 100);
    });
  });
}); 