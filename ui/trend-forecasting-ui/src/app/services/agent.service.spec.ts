import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { provideZoneChangeDetection } from '@angular/core';
import { AgentService } from './agent.service';
import { AgentRequest, AgentResponse } from '../models/agent.models';
import { environment } from '../../environments/environment';

describe('AgentService', () => {
  let service: AgentService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        AgentService,
        provideZoneChangeDetection()
      ]
    });
    service = TestBed.inject(AgentService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('askAgent', () => {
    it('should send POST request to agent/ask endpoint', () => {
      const mockRequest: AgentRequest = {
        query: 'How will AI trend next week?',
        session_id: 'test-session-123'
      };

      const mockResponse: AgentResponse = {
        text: 'Based on current trends, AI is expected to continue growing...',
        data: { forecast: 'positive' },
        metadata: { confidence: 0.85 },
        timestamp: '2024-01-01T00:00:00Z',
        request_id: 'req-123'
      };

      service.askAgent(mockRequest).subscribe(response => {
        expect(response).toEqual(mockResponse);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/agent/ask`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(mockRequest);
      req.flush(mockResponse);
    });

    it('should handle error responses gracefully', () => {
      const mockRequest: AgentRequest = {
        query: 'test query'
      };

      const errorMessage = 'Internal server error';

      service.askAgent(mockRequest).subscribe({
        next: () => fail('should have failed with 500 error'),
        error: (error) => {
          expect(error.status).toBe(500);
          expect(error.error).toBe(errorMessage);
        }
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/agent/ask`);
      req.flush(errorMessage, { status: 500, statusText: 'Internal Server Error' });
    });
  });

  describe('getHealth', () => {
    it('should send GET request to agent/health endpoint', () => {
      const mockHealthResponse = { status: 'healthy', timestamp: '2024-01-01T00:00:00Z' };

      service.getHealth().subscribe(response => {
        expect(response).toEqual(mockHealthResponse);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/agent/health`);
      expect(req.request.method).toBe('GET');
      req.flush(mockHealthResponse);
    });
  });

  describe('getCapabilities', () => {
    it('should send GET request to agent/capabilities endpoint', () => {
      const mockCapabilitiesResponse = {
        capabilities: ['trend_analysis', 'forecasting', 'data_visualization'],
        version: '1.0.0'
      };

      service.getCapabilities().subscribe(response => {
        expect(response).toEqual(mockCapabilitiesResponse);
      });

      const req = httpMock.expectOne(`${environment.apiUrl}/agent/capabilities`);
      expect(req.request.method).toBe('GET');
      req.flush(mockCapabilitiesResponse);
    });
  });
}); 