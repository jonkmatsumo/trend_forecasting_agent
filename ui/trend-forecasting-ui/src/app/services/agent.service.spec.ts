import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { AgentService } from './agent.service';
import { AgentRequest, AgentResponse, ChatMessage } from '../models/agent.models';

describe('AgentService', () => {
  let service: AgentService;
  let httpMock: HttpTestingController;
  const baseUrl = 'http://localhost:5000';

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [AgentService]
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

  describe('sendMessage', () => {
    const mockRequest: AgentRequest = {
      message: 'Test message',
      context: 'test-context',
      options: {
        maxTokens: 100,
        temperature: 0.7
      }
    };

    const mockResponse: AgentResponse = {
      response: 'Test response',
      messageId: 'msg-123',
      timestamp: new Date().toISOString(),
      metadata: {
        tokensUsed: 50,
        processingTime: 1000
      }
    };

    it('should send POST request to correct endpoint', () => {
      service.sendMessage(mockRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(mockRequest);
    });

    it('should return agent response on success', () => {
      service.sendMessage(mockRequest).subscribe(response => {
        expect(response).toEqual(mockResponse);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(mockResponse);
    });

    it('should handle HTTP errors', () => {
      const errorMessage = 'Server error';
      
      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(500);
          expect(error.error).toBe(errorMessage);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(errorMessage, { status: 500, statusText: 'Internal Server Error' });
    });

    it('should handle network errors', () => {
      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(0);
          expect(error.statusText).toBe('Unknown Error');
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.error(new ErrorEvent('Network error'));
    });

    it('should handle timeout errors', () => {
      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(408);
          expect(error.statusText).toBe('Request Timeout');
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush('Timeout', { status: 408, statusText: 'Request Timeout' });
    });
  });

  describe('getChatHistory', () => {
    const mockMessages: ChatMessage[] = [
      {
        id: 'msg-1',
        content: 'Hello',
        type: 'user',
        timestamp: new Date().toISOString()
      },
      {
        id: 'msg-2',
        content: 'Hi there!',
        type: 'agent',
        timestamp: new Date().toISOString()
      }
    ];

    it('should send GET request to correct endpoint', () => {
      service.getChatHistory().subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      expect(req.request.method).toBe('GET');
    });

    it('should return chat history on success', () => {
      service.getChatHistory().subscribe(messages => {
        expect(messages).toEqual(mockMessages);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      req.flush(mockMessages);
    });

    it('should handle empty chat history', () => {
      service.getChatHistory().subscribe(messages => {
        expect(messages).toEqual([]);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      req.flush([]);
    });

    it('should handle HTTP errors for chat history', () => {
      service.getChatHistory().subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(404);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      req.flush('Not found', { status: 404, statusText: 'Not Found' });
    });
  });

  describe('clearChatHistory', () => {
    it('should send DELETE request to correct endpoint', () => {
      service.clearChatHistory().subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      expect(req.request.method).toBe('DELETE');
    });

    it('should return success response', () => {
      const successResponse = { message: 'Chat history cleared successfully' };

      service.clearChatHistory().subscribe(response => {
        expect(response).toEqual(successResponse);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      req.flush(successResponse);
    });

    it('should handle HTTP errors for clear history', () => {
      service.clearChatHistory().subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(500);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat/history`);
      req.flush('Server error', { status: 500, statusText: 'Internal Server Error' });
    });
  });

  describe('getAgentStatus', () => {
    const mockStatus = {
      status: 'online',
      version: '1.0.0',
      uptime: 3600,
      lastActivity: new Date().toISOString()
    };

    it('should send GET request to correct endpoint', () => {
      service.getAgentStatus().subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/status`);
      expect(req.request.method).toBe('GET');
    });

    it('should return agent status on success', () => {
      service.getAgentStatus().subscribe(status => {
        expect(status).toEqual(mockStatus);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/status`);
      req.flush(mockStatus);
    });

    it('should handle HTTP errors for status', () => {
      service.getAgentStatus().subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(503);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/status`);
      req.flush('Service unavailable', { status: 503, statusText: 'Service Unavailable' });
    });
  });

  describe('getAgentConfig', () => {
    const mockConfig = {
      model: 'gpt-4',
      maxTokens: 1000,
      temperature: 0.7,
      systemPrompt: 'You are a helpful assistant'
    };

    it('should send GET request to correct endpoint', () => {
      service.getAgentConfig().subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      expect(req.request.method).toBe('GET');
    });

    it('should return agent config on success', () => {
      service.getAgentConfig().subscribe(config => {
        expect(config).toEqual(mockConfig);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      req.flush(mockConfig);
    });

    it('should handle HTTP errors for config', () => {
      service.getAgentConfig().subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(403);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      req.flush('Forbidden', { status: 403, statusText: 'Forbidden' });
    });
  });

  describe('updateAgentConfig', () => {
    const mockConfig = {
      model: 'gpt-4',
      maxTokens: 1500,
      temperature: 0.8
    };

    it('should send PUT request to correct endpoint', () => {
      service.updateAgentConfig(mockConfig).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      expect(req.request.method).toBe('PUT');
      expect(req.request.body).toEqual(mockConfig);
    });

    it('should return updated config on success', () => {
      const updatedConfig = { ...mockConfig, updated: true };

      service.updateAgentConfig(mockConfig).subscribe(config => {
        expect(config).toEqual(updatedConfig);
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      req.flush(updatedConfig);
    });

    it('should handle HTTP errors for config update', () => {
      service.updateAgentConfig(mockConfig).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error.status).toBe(400);
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/config`);
      req.flush('Bad request', { status: 400, statusText: 'Bad Request' });
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed JSON responses', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error).toBeTruthy();
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush('Invalid JSON', { status: 200, statusText: 'OK' });
    });

    it('should handle null responses', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error).toBeTruthy();
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(null);
    });

    it('should handle undefined responses', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      service.sendMessage(mockRequest).subscribe({
        next: () => fail('should have failed'),
        error: (error) => {
          expect(error).toBeTruthy();
        }
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(null);
    });
  });

  describe('Request Validation', () => {
    it('should handle empty message', () => {
      const mockRequest: AgentRequest = {
        message: '',
        context: 'test-context'
      };

      service.sendMessage(mockRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      expect(req.request.body).toEqual(mockRequest);
      req.flush({ response: 'Empty message handled' });
    });

    it('should handle request without context', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message'
      };

      service.sendMessage(mockRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      expect(req.request.body).toEqual(mockRequest);
      req.flush({ response: 'No context handled' });
    });

    it('should handle request without options', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      service.sendMessage(mockRequest).subscribe();

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      expect(req.request.body).toEqual(mockRequest);
      req.flush({ response: 'No options handled' });
    });
  });

  describe('Response Processing', () => {
    it('should handle response with missing fields', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      const incompleteResponse = {
        response: 'Test response'
        // Missing other fields
      };

      service.sendMessage(mockRequest).subscribe(response => {
        expect(response.response).toBe('Test response');
        expect(response.messageId).toBeUndefined();
        expect(response.timestamp).toBeUndefined();
        expect(response.metadata).toBeUndefined();
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(incompleteResponse);
    });

    it('should handle response with extra fields', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      const extendedResponse = {
        response: 'Test response',
        messageId: 'msg-123',
        timestamp: new Date().toISOString(),
        metadata: {
          tokensUsed: 50,
          processingTime: 1000
        },
        extraField: 'should be ignored'
      };

      service.sendMessage(mockRequest).subscribe(response => {
        expect(response.response).toBe('Test response');
        expect(response.messageId).toBe('msg-123');
        expect(response.timestamp).toBe(extendedResponse.timestamp);
        expect(response.metadata).toEqual(extendedResponse.metadata);
        expect((response as any).extraField).toBeUndefined();
      });

      const req = httpMock.expectOne(`${baseUrl}/api/agent/chat`);
      req.flush(extendedResponse);
    });
  });

  describe('Performance', () => {
    it('should handle multiple concurrent requests', () => {
      const mockRequest: AgentRequest = {
        message: 'Test message',
        context: 'test-context'
      };

      const responses: AgentResponse[] = [];

      // Send multiple concurrent requests
      for (let i = 0; i < 5; i++) {
        service.sendMessage(mockRequest).subscribe(response => {
          responses.push(response);
        });
      }

      // Verify all requests were made
      const requests = httpMock.match(`${baseUrl}/api/agent/chat`);
      expect(requests.length).toBe(5);

      // Respond to all requests
      requests.forEach((req, index) => {
        req.flush({
          response: `Response ${index + 1}`,
          messageId: `msg-${index + 1}`,
          timestamp: new Date().toISOString()
        });
      });

      expect(responses.length).toBe(5);
    });
  });
}); 