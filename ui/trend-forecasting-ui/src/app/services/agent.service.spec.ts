import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { AgentService } from './agent.service';
import { AgentRequest, AgentResponse, AgentConfig, ChatMessage } from '../models/agent.models';
import { environment } from '../../environments/environment';
import { ErrorHandlerService } from './error-handler.service';
import { LoadingService } from './loading.service';

describe('AgentService', () => {
  let service: AgentService;
  let httpMock: HttpTestingController;
  let errorHandler: jasmine.SpyObj<ErrorHandlerService>;
  let loadingService: jasmine.SpyObj<LoadingService>;

  beforeEach(() => {
    const errorHandlerSpy = jasmine.createSpyObj('ErrorHandlerService', ['createRetryableRequest']);
    const loadingServiceSpy = jasmine.createSpyObj('LoadingService', ['withLoading']);

    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        AgentService,
        { provide: ErrorHandlerService, useValue: errorHandlerSpy },
        { provide: LoadingService, useValue: loadingServiceSpy }
      ]
    });

    service = TestBed.inject(AgentService);
    httpMock = TestBed.inject(HttpTestingController);
    errorHandler = TestBed.inject(ErrorHandlerService) as jasmine.SpyObj<ErrorHandlerService>;
    loadingService = TestBed.inject(LoadingService) as jasmine.SpyObj<LoadingService>;
  });

  afterEach(() => {
    if (httpMock) {
      httpMock.verify();
    }
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('sendMessage', () => {
    it('should send a message and return response', () => {
      const request: AgentRequest = { message: 'Hello agent' };
      const mockResponse: AgentResponse = {
        response: 'Hello! How can I help you?',
        messageId: 'msg-123',
        timestamp: new Date().toISOString()
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockResponse);
          observer.complete();
        }
      } as any);

      service.sendMessage(request).subscribe(response => {
        expect(response).toEqual(mockResponse);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('askAgent', () => {
    it('should ask agent and return response', () => {
      const request: AgentRequest = { message: 'What is the weather?' };
      const mockResponse: AgentResponse = {
        response: 'I cannot provide weather information.',
        messageId: 'msg-456',
        timestamp: new Date().toISOString()
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockResponse);
          observer.complete();
        }
      } as any);

      service.askAgent(request).subscribe(response => {
        expect(response).toEqual(mockResponse);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('getHealth', () => {
    it('should get health status', () => {
      const mockHealth = { status: 'healthy', timestamp: new Date().toISOString() };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockHealth);
          observer.complete();
        }
      } as any);

      service.getHealth().subscribe(health => {
        expect(health).toEqual(mockHealth);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('getCapabilities', () => {
    it('should get agent capabilities', () => {
      const mockCapabilities = {
        features: ['trend_analysis', 'forecasting'],
        models: ['gpt-4', 'gpt-3.5-turbo']
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockCapabilities);
          observer.complete();
        }
      } as any);

      service.getCapabilities().subscribe(capabilities => {
        expect(capabilities).toEqual(mockCapabilities);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('getAgentConfig', () => {
    it('should get agent configuration', () => {
      const mockConfig: AgentConfig = {
        model: 'gpt-4',
        temperature: 0.7,
        maxTokens: 1000
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockConfig);
          observer.complete();
        }
      } as any);

      service.getAgentConfig().subscribe(config => {
        expect(config).toEqual(mockConfig);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('updateAgentConfig', () => {
    it('should update agent configuration', () => {
      const config: AgentConfig = {
        model: 'gpt-4',
        temperature: 0.8,
        maxTokens: 1500
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(config);
          observer.complete();
        }
      } as any);

      service.updateAgentConfig(config).subscribe(updatedConfig => {
        expect(updatedConfig).toEqual(config);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('getChatHistory', () => {
    it('should get chat history', () => {
      const mockHistory: ChatMessage[] = [
        {
          id: 1,
          type: 'user',
          text: 'Hello agent',
          timestamp: new Date()
        },
        {
          id: 2,
          type: 'agent',
          text: 'Hello! How can I help you?',
          timestamp: new Date()
        }
      ];

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockHistory);
          observer.complete();
        }
      } as any);

      service.getChatHistory().subscribe(history => {
        expect(history).toEqual(mockHistory);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('clearChatHistory', () => {
    it('should clear chat history', () => {
      const mockResponse = { message: 'Chat history cleared' };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockResponse);
          observer.complete();
        }
      } as any);

      service.clearChatHistory().subscribe(response => {
        expect(response).toEqual(mockResponse);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });

  describe('getAgentStatus', () => {
    it('should get agent status', () => {
      const mockStatus = {
        status: 'active',
        uptime: 3600,
        requestsProcessed: 100
      };

      errorHandler.createRetryableRequest.and.returnValue({
        subscribe: (observer: any) => {
          observer.next(mockStatus);
          observer.complete();
        }
      } as any);

      service.getAgentStatus().subscribe(status => {
        expect(status).toEqual(mockStatus);
      });

      expect(errorHandler.createRetryableRequest).toHaveBeenCalled();
    });
  });
}); 