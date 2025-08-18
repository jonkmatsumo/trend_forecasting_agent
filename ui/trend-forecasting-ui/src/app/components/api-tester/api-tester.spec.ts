import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { provideZonelessChangeDetection } from '@angular/core';
import { ApiTesterComponent } from './api-tester';
import { ApiTestService } from '../../services/api-test.service';
import { ApiEndpoint } from '../../models/api.models';
import { of } from 'rxjs';

describe('ApiTesterComponent', () => {
  let component: ApiTesterComponent;
  let fixture: ComponentFixture<ApiTesterComponent>;
  let apiTestService: jasmine.SpyObj<ApiTestService>;

  beforeEach(async () => {
    const spy = jasmine.createSpyObj('ApiTestService', [
      'testAllEndpoints',
      'testCorsConfiguration',
      'testProxyConfiguration'
    ]);

    await TestBed.configureTestingModule({
      imports: [
        ApiTesterComponent,
        HttpClientTestingModule,
        NoopAnimationsModule
      ],
      providers: [
        { provide: ApiTestService, useValue: spy },
        provideZonelessChangeDetection()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ApiTesterComponent);
    component = fixture.componentInstance;
    apiTestService = TestBed.inject(ApiTestService) as jasmine.SpyObj<ApiTestService>;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize with default values', () => {
    expect(component.selectedEndpoint).toBeNull();
    expect(component.response).toBeNull();
    expect(component.isLoading).toBe(false);
    expect(component.error).toBe('');
    expect(component.integrationTestResults).toBeUndefined();
    expect(component.corsTestResult).toBeUndefined();
    expect(component.proxyTestResults).toBeUndefined();
    expect(component.isRunningIntegrationTests).toBe(false);
  });

  it('should have endpoint options', () => {
    expect(component.endpoints).toBeDefined();
    expect(component.endpoints.length).toBeGreaterThan(0);
    expect(component.endpoints[0].name).toBeDefined();
    expect(component.endpoints[0].method).toBeDefined();
    expect(component.endpoints[0].path).toBeDefined();
  });

  describe('runIntegrationTests', () => {
    it('should run integration tests successfully', () => {
      const mockResults = {
        name: 'Full API Test Suite',
        description: 'Comprehensive API test suite with 6 endpoints',
        tests: [
          { endpoint: '/health', method: 'GET', status: 'success' as const, responseTime: 100 },
          { endpoint: '/api/trends', method: 'GET', status: 'success' as const, responseTime: 150 },
          { endpoint: '/agent/ask', method: 'POST', status: 'error' as const, responseTime: 0 }
        ],
        summary: {
          total: 3,
          successful: 2,
          failed: 1,
          averageResponseTime: 83
        }
      };

      apiTestService.testAllEndpoints.and.returnValue(of(mockResults));

      component.runIntegrationTests();

      expect(component.isRunningIntegrationTests).toBe(true);
      expect(apiTestService.testAllEndpoints).toHaveBeenCalled();
    });
  });

  describe('testCorsConfiguration', () => {
    it('should test CORS configuration successfully', () => {
      const mockResult = {
        endpoint: 'CORS Test',
        method: 'GET',
        status: 'success' as const,
        statusCode: 200,
        responseTime: 100,
        data: {
          corsHeaders: {
            'access-control-allow-origin': 'http://localhost:4200'
          }
        }
      };

      apiTestService.testCorsConfiguration.and.returnValue(of(mockResult));

      component.testCorsConfiguration();

      expect(apiTestService.testCorsConfiguration).toHaveBeenCalled();
    });
  });

  describe('testProxyConfiguration', () => {
    it('should test proxy configuration successfully', () => {
      const mockResults = [
        {
          endpoint: '/health',
          method: 'GET',
          status: 'success' as const,
          responseTime: 100
        },
        {
          endpoint: '/agent/health',
          method: 'GET',
          status: 'success' as const,
          responseTime: 150
        }
      ];

      apiTestService.testProxyConfiguration.and.returnValue(of(mockResults));

      component.testProxyConfiguration();

      expect(apiTestService.testProxyConfiguration).toHaveBeenCalled();
    });
  });

  describe('getStatusColor', () => {
    it('should return correct color for success status', () => {
      expect(component.getStatusColor('success')).toBe('green');
    });

    it('should return correct color for error status', () => {
      expect(component.getStatusColor('error')).toBe('red');
    });

    it('should return correct color for warning status', () => {
      expect(component.getStatusColor('warning')).toBe('orange');
    });

    it('should return default color for unknown status', () => {
      expect(component.getStatusColor('unknown')).toBe('gray');
    });
  });

  describe('getStatusIcon', () => {
    it('should return correct icon for success status', () => {
      expect(component.getStatusIcon('success')).toBe('check_circle');
    });

    it('should return correct icon for error status', () => {
      expect(component.getStatusIcon('error')).toBe('error');
    });

    it('should return correct icon for warning status', () => {
      expect(component.getStatusIcon('warning')).toBe('warning');
    });

    it('should return default icon for unknown status', () => {
      expect(component.getStatusIcon('unknown')).toBe('help');
    });
  });

  describe('template rendering', () => {
    it('should display the component', () => {
      const element = fixture.nativeElement;
      expect(element).toBeTruthy();
    });

    it('should have endpoint selection', () => {
      const element = fixture.nativeElement;
      expect(element.querySelector('mat-form-field')).toBeTruthy();
    });

    it('should have send button', () => {
      const element = fixture.nativeElement;
      expect(element.querySelector('button')).toBeTruthy();
    });
  });
}); 