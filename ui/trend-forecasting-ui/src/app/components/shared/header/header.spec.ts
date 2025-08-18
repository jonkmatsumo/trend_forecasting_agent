import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { RouterTestingModule } from '@angular/router/testing';
import { Router } from '@angular/router';
import { provideZonelessChangeDetection } from '@angular/core';
import { HeaderComponent } from './header';

describe('HeaderComponent', () => {
  let component: HeaderComponent;
  let fixture: ComponentFixture<HeaderComponent>;
  let router: Router;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        HeaderComponent,
        NoopAnimationsModule,
        RouterTestingModule.withRoutes([
          { path: 'agent-chat', component: HeaderComponent },
          { path: 'api-tester', component: HeaderComponent }
        ])
      ],
      providers: [
        provideZonelessChangeDetection()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HeaderComponent);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('isActive', () => {
    it('should return true for active route', async () => {
      await router.navigate(['/agent-chat']);
      
      const result = component.isActive('/agent-chat');
      
      expect(result).toBe(true);
    });

    it('should return false for inactive route', async () => {
      await router.navigate(['/agent-chat']);
      
      const result = component.isActive('/api-tester');
      
      expect(result).toBe(false);
    });

    it('should return false for non-existent route', async () => {
      await router.navigate(['/agent-chat']);
      
      const result = component.isActive('/non-existent');
      
      expect(result).toBe(false);
    });
  });

  describe('template rendering', () => {
    it('should display header element', () => {
      const headerElement = fixture.nativeElement.querySelector('mat-toolbar');
      expect(headerElement).toBeTruthy();
    });

    it('should have proper Material Design classes', () => {
      const toolbarElement = fixture.nativeElement.querySelector('mat-toolbar');
      expect(toolbarElement.classList.contains('mat-toolbar')).toBe(true);
    });
  });
}); 