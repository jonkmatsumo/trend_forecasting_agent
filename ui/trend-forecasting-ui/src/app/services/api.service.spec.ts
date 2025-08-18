import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { ApiService } from './api.service';
import { ApiRequest, ApiResponse } from '../models/api.models';
import { environment } from '../../environments/environment';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        ApiService,
        provideZonelessChangeDetection()
      ]
    });
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('sendRequest', () => {
    it('should send GET request successfully', () => {
      const mockResponse = { status: 'ok', message: 'Health check passed' };
      const request: ApiRequest = {
        method: 'GET',
        url: `${environment.apiUrl}/health`,
        headers: {},
        body: null
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(200);
        expect(response.data).toEqual(mockResponse);
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      expect(req.request.method).toBe('GET');
      req.flush(mockResponse);
    });

    it('should send POST request successfully', () => {
      const mockRequest = { query: 'test query' };
      const mockResponse = { response: 'test response' };
      const request: ApiRequest = {
        method: 'POST',
        url: `${environment.apiUrl}/agent/ask`,
        headers: { 'Content-Type': 'application/json' },
        body: mockRequest
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(200);
        expect(response.data).toEqual(mockResponse);
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/agent/ask`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(mockRequest);
      req.flush(mockResponse);
    });

    it('should handle HTTP errors gracefully', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${environment.apiUrl}/health`,
        headers: {},
        body: null
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(500);
        expect(response.statusText).toBe('Internal Server Error');
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.flush('Internal server error', { 
        status: 500, 
        statusText: 'Internal Server Error' 
      });
    });

    it('should handle network errors', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${environment.apiUrl}/health`,
        headers: {},
        body: null
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(0);
        expect(response.statusText).toBe('Unknown Error');
        expect(response.responseTime).toBeGreaterThan(0);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      req.error(new ErrorEvent('Network error'));
    });

    it('should throw error for unsupported HTTP method', () => {
      const request: ApiRequest = {
        method: 'PATCH' as any,
        url: `${environment.apiUrl}/test`,
        headers: {},
        body: null
      };

      expect(() => {
        service.sendRequest(request).subscribe();
      }).toThrow('Unsupported HTTP method: PATCH');
    });

    it('should include custom headers in request', () => {
      const request: ApiRequest = {
        method: 'GET',
        url: `${environment.apiUrl}/health`,
        headers: { 'Authorization': 'Bearer token123' },
        body: null
      };

      service.sendRequest(request).subscribe();

      const req = httpMock.expectOne(`${environment.apiUrl}/health`);
      expect(req.request.headers.get('Authorization')).toBe('Bearer token123');
      req.flush({});
    });

    it('should handle PUT request', () => {
      const mockData = { name: 'updated', value: 456 };
      const mockResponse = { id: 1, ...mockData };
      const request: ApiRequest = {
        method: 'PUT',
        url: `${environment.apiUrl}/test/1`,
        headers: { 'Content-Type': 'application/json' },
        body: mockData
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(200);
        expect(response.data).toEqual(mockResponse);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/test/1`);
      expect(req.request.method).toBe('PUT');
      expect(req.request.body).toEqual(mockData);
      req.flush(mockResponse);
    });

    it('should handle DELETE request', () => {
      const mockResponse = { status: 'deleted' };
      const request: ApiRequest = {
        method: 'DELETE',
        url: `${environment.apiUrl}/test/1`,
        headers: {},
        body: null
      };

      service.sendRequest(request).subscribe((response: ApiResponse) => {
        expect(response.status).toBe(200);
        expect(response.data).toEqual(mockResponse);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/test/1`);
      expect(req.request.method).toBe('DELETE');
      req.flush(mockResponse);
    });
  });
}); 