import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { NotificationComponent, NotificationAction } from './notification';

describe('NotificationComponent', () => {
  let component: NotificationComponent;
  let fixture: ComponentFixture<NotificationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        NoopAnimationsModule,
        MatIconModule,
        MatButtonModule,
        NotificationComponent
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(NotificationComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('Input Properties', () => {
    it('should have default values', () => {
      expect(component.type).toBe('info');
      expect(component.message).toBe('');
      expect(component.autoClose).toBe(true);
      expect(component.dismissible).toBe(true);
      expect(component.actions).toBeUndefined();
    });

    it('should accept custom input values', () => {
      component.type = 'success';
      component.title = 'Test Title';
      component.message = 'Test Message';
      component.duration = 5000;
      component.autoClose = false;
      component.dismissible = false;

      expect(component.type).toBe('success');
      expect(component.title).toBe('Test Title');
      expect(component.message).toBe('Test Message');
      expect(component.duration).toBe(5000);
      expect(component.autoClose).toBe(false);
      expect(component.dismissible).toBe(false);
    });
  });

  describe('Rendering', () => {
    it('should render notification with basic content', () => {
      component.message = 'Test notification message';
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent).toContain('Test notification message');
    });

    it('should render notification with title', () => {
      component.title = 'Test Title';
      component.message = 'Test Message';
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent).toContain('Test Title');
      expect(element.textContent).toContain('Test Message');
    });

    it('should render notification without title when not provided', () => {
      component.message = 'Test Message';
      fixture.detectChanges();

      const element = fixture.nativeElement;
      const titleElement = element.querySelector('.notification-title');
      expect(titleElement).toBeNull();
    });

    it('should apply correct CSS class based on type', () => {
      component.type = 'success';
      component.message = 'Test message';
      fixture.detectChanges();

      const container = fixture.nativeElement.querySelector('.notification-container');
      expect(container.classList).toContain('notification--success');
    });

    it('should render different types correctly', () => {
      const types = ['success', 'error', 'warning', 'info'];
      
      types.forEach(type => {
        component.type = type as any;
        component.message = `Test ${type} message`;
        fixture.detectChanges();

        const container = fixture.nativeElement.querySelector('.notification-container');
        expect(container.classList).toContain(`notification--${type}`);
      });
    });
  });

  describe('Icon and Color', () => {
    it('should return correct icon for each type', () => {
      const expectedIcons = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning',
        info: 'info'
      };

      Object.entries(expectedIcons).forEach(([type, expectedIcon]) => {
        component.type = type as any;
        expect(component.getIcon()).toBe(expectedIcon);
      });
    });

    it('should return correct icon color for each type', () => {
      const expectedColors = {
        success: 'primary',
        error: 'warn',
        warning: 'accent',
        info: 'primary'
      };

      Object.entries(expectedColors).forEach(([type, expectedColor]) => {
        component.type = type as any;
        expect(component.getIconColor()).toBe(expectedColor);
      });
    });

    it('should render icon with correct color', () => {
      component.type = 'success';
      component.message = 'Test message';
      fixture.detectChanges();

      const iconElement = fixture.nativeElement.querySelector('mat-icon');
      expect(iconElement).toBeTruthy();
      expect(iconElement.getAttribute('color')).toBe('primary');
    });
  });

  describe('Actions', () => {
    it('should render action buttons when provided', () => {
      const actions: NotificationAction[] = [
        { label: 'Action 1', color: 'primary' },
        { label: 'Action 2', color: 'accent' }
      ];

      component.actions = actions;
      component.message = 'Test message';
      fixture.detectChanges();

      const actionButtons = fixture.nativeElement.querySelectorAll('.notification-actions button');
      expect(actionButtons.length).toBe(2);
      expect(actionButtons[0].textContent).toContain('Action 1');
      expect(actionButtons[1].textContent).toContain('Action 2');
    });

    it('should not render action buttons when not provided', () => {
      component.message = 'Test message';
      fixture.detectChanges();

      const actionButtons = fixture.nativeElement.querySelector('.notification-actions');
      expect(actionButtons).toBeNull();
    });

    it('should handle action button clicks', () => {
      const actionSpy = jasmine.createSpy('action');
      const actions: NotificationAction[] = [
        { label: 'Test Action', color: 'primary', action: actionSpy }
      ];

      component.actions = actions;
      component.message = 'Test message';
      fixture.detectChanges();

      const actionButton = fixture.nativeElement.querySelector('.notification-actions button');
      actionButton.click();

      expect(actionSpy).toHaveBeenCalled();
    });

    it('should emit actionClicked event when action is clicked', () => {
      const actions: NotificationAction[] = [
        { label: 'Test Action', color: 'primary' }
      ];

      component.actions = actions;
      component.message = 'Test message';
      fixture.detectChanges();

      spyOn(component.actionClicked, 'emit');

      const actionButton = fixture.nativeElement.querySelector('.notification-actions button');
      actionButton.click();

      expect(component.actionClicked.emit).toHaveBeenCalledWith(actions[0]);
    });
  });

  describe('Close Button', () => {
    it('should render close button when dismissible is true', () => {
      component.dismissible = true;
      component.message = 'Test message';
      fixture.detectChanges();

      const closeButton = fixture.nativeElement.querySelector('.close-button');
      expect(closeButton).toBeTruthy();
    });

    it('should not render close button when dismissible is false', () => {
      component.dismissible = false;
      component.message = 'Test message';
      fixture.detectChanges();

      const closeButton = fixture.nativeElement.querySelector('.close-button');
      expect(closeButton).toBeNull();
    });

    it('should call dismiss when close button is clicked', () => {
      component.dismissible = true;
      component.message = 'Test message';
      fixture.detectChanges();

      spyOn(component, 'dismiss');

      const closeButton = fixture.nativeElement.querySelector('.close-button');
      closeButton.click();

      expect(component.dismiss).toHaveBeenCalled();
    });
  });

  describe('Progress Bar', () => {
    it('should render progress bar when autoClose and duration are provided', () => {
      component.autoClose = true;
      component.duration = 5000;
      component.message = 'Test message';
      fixture.detectChanges();

      const progressBar = fixture.nativeElement.querySelector('.notification-progress');
      expect(progressBar).toBeTruthy();
    });

    it('should not render progress bar when autoClose is false', () => {
      component.autoClose = false;
      component.duration = 5000;
      component.message = 'Test message';
      fixture.detectChanges();

      const progressBar = fixture.nativeElement.querySelector('.notification-progress');
      expect(progressBar).toBeNull();
    });

    it('should not render progress bar when duration is not provided', () => {
      component.autoClose = true;
      component.duration = undefined;
      component.message = 'Test message';
      fixture.detectChanges();

      const progressBar = fixture.nativeElement.querySelector('.notification-progress');
      expect(progressBar).toBeNull();
    });

    it('should set correct animation duration on progress bar', () => {
      component.autoClose = true;
      component.duration = 3000;
      component.message = 'Test message';
      fixture.detectChanges();

      const progressBar = fixture.nativeElement.querySelector('.progress-bar');
      expect(progressBar.style.animationDuration).toBe('3000ms');
    });
  });

  describe('Animation States', () => {
    it('should start with void animation state', () => {
      expect(component.animationState).toBe('void');
    });

    it('should change to visible state after initialization', fakeAsync(() => {
      component.message = 'Test message';
      fixture.detectChanges();

      tick(100); // Wait for setTimeout in ngOnInit

      expect(component.animationState).toBe('visible');
    }));

    it('should change to hidden state when dismissed', fakeAsync(() => {
      component.message = 'Test message';
      fixture.detectChanges();

      tick(100); // Wait for setTimeout in ngOnInit
      expect(component.animationState).toBe('visible');

      component.dismiss();
      expect(component.animationState).toBe('hidden');
    }));
  });

  describe('Auto-close Functionality', () => {
    beforeEach(() => {
      jasmine.clock().install();
    });

    afterEach(() => {
      jasmine.clock().uninstall();
    });

    it('should auto-dismiss after duration when autoClose is true', fakeAsync(() => {
      component.autoClose = true;
      component.duration = 1000;
      component.message = 'Test message';
      
      spyOn(component, 'dismiss');
      
      fixture.detectChanges();
      tick(100); // Wait for setTimeout in ngOnInit

      jasmine.clock().tick(1000); // Wait for auto-close timeout
      tick(1000);

      expect(component.dismiss).toHaveBeenCalled();
    }));

    it('should not auto-dismiss when autoClose is false', fakeAsync(() => {
      component.autoClose = false;
      component.duration = 1000;
      component.message = 'Test message';
      
      spyOn(component, 'dismiss');
      
      fixture.detectChanges();
      tick(100); // Wait for setTimeout in ngOnInit

      jasmine.clock().tick(1000); // Wait for auto-close timeout
      tick(1000);

      expect(component.dismiss).not.toHaveBeenCalled();
    }));

    it('should clear timeout when dismissed manually', fakeAsync(() => {
      component.autoClose = true;
      component.duration = 1000;
      component.message = 'Test message';
      
      fixture.detectChanges();
      tick(100); // Wait for setTimeout in ngOnInit

      component.dismiss();

      jasmine.clock().tick(1000); // Wait for auto-close timeout
      tick(1000);

      // Should not call dismiss again since it was already dismissed
      expect(component.animationState).toBe('hidden');
    }));
  });

  describe('Event Emissions', () => {
    it('should emit dismissed event when animation completes and state is hidden', () => {
      component.message = 'Test message';
      fixture.detectChanges();

      spyOn(component.dismissed, 'emit');

      component.animationState = 'hidden';
      component.onAnimationComplete();

      expect(component.dismissed.emit).toHaveBeenCalled();
    });

    it('should not emit dismissed event when animation completes and state is not hidden', () => {
      component.message = 'Test message';
      fixture.detectChanges();

      spyOn(component.dismissed, 'emit');

      component.animationState = 'visible';
      component.onAnimationComplete();

      expect(component.dismissed.emit).not.toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA label on close button', () => {
      component.dismissible = true;
      component.message = 'Test message';
      fixture.detectChanges();

      const closeButton = fixture.nativeElement.querySelector('.close-button');
      expect(closeButton.getAttribute('aria-label')).toBe('Close notification');
    });

    it('should have proper button structure for actions', () => {
      const actions: NotificationAction[] = [
        { label: 'Test Action', color: 'primary' }
      ];

      component.actions = actions;
      component.message = 'Test message';
      fixture.detectChanges();

      const actionButton = fixture.nativeElement.querySelector('.notification-actions button');
      expect(actionButton).toBeTruthy();
      expect(actionButton.textContent.trim()).toBe('Test Action');
    });
  });

  describe('Lifecycle', () => {
    it('should clear timeout on destroy', fakeAsync(() => {
      component.autoClose = true;
      component.duration = 1000;
      component.message = 'Test message';
      
      fixture.detectChanges();
      tick(100); // Wait for setTimeout in ngOnInit

      spyOn(window, 'clearTimeout');
      
      component.ngOnDestroy();

      expect(window.clearTimeout).toHaveBeenCalled();
    }));

    it('should not throw error when destroyed without timeout', () => {
      component.message = 'Test message';
      fixture.detectChanges();

      expect(() => component.ngOnDestroy()).not.toThrow();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty message', () => {
      component.message = '';
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent.trim()).toBe('');
    });

    it('should handle very long messages', () => {
      const longMessage = 'A'.repeat(1000);
      component.message = longMessage;
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent).toContain(longMessage);
    });

    it('should handle special characters in message', () => {
      const specialMessage = 'Test message with special chars: !@#$%^&*()_+-=[]{}|;:,.<>?';
      component.message = specialMessage;
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent).toContain(specialMessage);
    });

    it('should handle HTML in message (should be escaped)', () => {
      const htmlMessage = '<script>alert("test")</script>';
      component.message = htmlMessage;
      fixture.detectChanges();

      const element = fixture.nativeElement;
      expect(element.textContent).toContain(htmlMessage);
      // Should not execute script
      expect(element.querySelector('script')).toBeNull();
    });
  });
}); 