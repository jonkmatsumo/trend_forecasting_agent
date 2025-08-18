import { TestBed } from '@angular/core/testing';
import { NotificationService, NotificationConfig, NotificationInstance } from './notification.service';

describe('NotificationService', () => {
  let service: NotificationService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [NotificationService]
    });
    service = TestBed.inject(NotificationService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('Basic Notification Methods', () => {
    it('should show success notification', () => {
      const message = 'Test success message';
      const title = 'Success Title';
      
      const id = service.success(message, title);
      
      expect(id).toBeTruthy();
      expect(typeof id).toBe('string');
      
      const notifications = service.currentNotifications;
      expect(notifications.length).toBe(1);
      expect(notifications[0].type).toBe('success');
      expect(notifications[0].message).toBe(message);
      expect(notifications[0].title).toBe(title);
      expect(notifications[0].autoClose).toBe(true);
      expect(notifications[0].dismissible).toBe(true);
    });

    it('should show error notification', () => {
      const message = 'Test error message';
      const title = 'Error Title';
      
      const id = service.error(message, title);
      
      expect(id).toBeTruthy();
      
      const notifications = service.currentNotifications;
      expect(notifications.length).toBe(1);
      expect(notifications[0].type).toBe('error');
      expect(notifications[0].message).toBe(message);
      expect(notifications[0].title).toBe(title);
    });

    it('should show warning notification', () => {
      const message = 'Test warning message';
      const title = 'Warning Title';
      
      const id = service.warning(message, title);
      
      expect(id).toBeTruthy();
      
      const notifications = service.currentNotifications;
      expect(notifications.length).toBe(1);
      expect(notifications[0].type).toBe('warning');
      expect(notifications[0].message).toBe(message);
      expect(notifications[0].title).toBe(title);
    });

    it('should show info notification', () => {
      const message = 'Test info message';
      const title = 'Info Title';
      
      const id = service.info(message, title);
      
      expect(id).toBeTruthy();
      
      const notifications = service.currentNotifications;
      expect(notifications.length).toBe(1);
      expect(notifications[0].type).toBe('info');
      expect(notifications[0].message).toBe(message);
      expect(notifications[0].title).toBe(title);
    });
  });

  describe('Custom Notification Configuration', () => {
    it('should show custom notification with full config', () => {
      const config: NotificationConfig = {
        type: 'success',
        title: 'Custom Title',
        message: 'Custom message',
        duration: 10000,
        autoClose: false,
        dismissible: false,
        actions: [
          {
            label: 'Action 1',
            color: 'primary',
            action: () => console.log('Action 1')
          }
        ]
      };
      
      const id = service.show(config);
      
      expect(id).toBeTruthy();
      
      const notifications = service.currentNotifications;
      expect(notifications.length).toBe(1);
      expect(notifications[0].type).toBe(config.type);
      expect(notifications[0].title).toBe(config.title);
      expect(notifications[0].message).toBe(config.message);
      expect(notifications[0].duration).toBe(config.duration);
      expect(notifications[0].autoClose).toBe(config.autoClose);
      expect(notifications[0].dismissible).toBe(config.dismissible);
      expect(notifications[0].actions).toEqual(config.actions);
    });

    it('should generate unique IDs for notifications', () => {
      const id1 = service.success('Message 1');
      const id2 = service.success('Message 2');
      
      expect(id1).not.toBe(id2);
      expect(typeof id1).toBe('string');
      expect(typeof id2).toBe('string');
    });
  });

  describe('Notification Management', () => {
    beforeEach(() => {
      service.clear();
    });

    it('should remove specific notification', () => {
      const id1 = service.success('Message 1');
      const id2 = service.error('Message 2');
      
      expect(service.currentNotifications.length).toBe(2);
      
      service.remove(id1);
      
      expect(service.currentNotifications.length).toBe(1);
      expect(service.currentNotifications[0].id).toBe(id2);
    });

    it('should clear all notifications', () => {
      service.success('Message 1');
      service.error('Message 2');
      service.warning('Message 3');
      
      expect(service.currentNotifications.length).toBe(3);
      
      service.clear();
      
      expect(service.currentNotifications.length).toBe(0);
    });

    it('should clear notifications by type', () => {
      service.success('Success 1');
      service.success('Success 2');
      service.error('Error 1');
      service.warning('Warning 1');
      
      expect(service.currentNotifications.length).toBe(4);
      
      service.clearByType('success');
      
      expect(service.currentNotifications.length).toBe(2);
      expect(service.currentNotifications.every(n => n.type !== 'success')).toBe(true);
    });

    it('should update notification', () => {
      const id = service.success('Original message');
      
      service.update(id, {
        message: 'Updated message',
        type: 'error'
      });
      
      const notification = service.getById(id);
      expect(notification?.message).toBe('Updated message');
      expect(notification?.type).toBe('error');
    });

    it('should get notification by ID', () => {
      const id = service.success('Test message');
      
      const notification = service.getById(id);
      
      expect(notification).toBeTruthy();
      expect(notification?.id).toBe(id);
      expect(notification?.message).toBe('Test message');
    });

    it('should check if notification exists', () => {
      const id = service.success('Test message');
      
      expect(service.exists(id)).toBe(true);
      expect(service.exists('non-existent-id')).toBe(false);
    });
  });

  describe('Notification Counts', () => {
    beforeEach(() => {
      service.clear();
    });

    it('should get notification count', () => {
      expect(service.count).toBe(0);
      
      service.success('Message 1');
      expect(service.count).toBe(1);
      
      service.error('Message 2');
      expect(service.count).toBe(2);
    });

    it('should get count by type', () => {
      service.success('Success 1');
      service.success('Success 2');
      service.error('Error 1');
      service.warning('Warning 1');
      
      expect(service.getCountByType('success')).toBe(2);
      expect(service.getCountByType('error')).toBe(1);
      expect(service.getCountByType('warning')).toBe(1);
      expect(service.getCountByType('info')).toBe(0);
    });
  });

  describe('Advanced Features', () => {
    beforeEach(() => {
      service.clear();
    });

    it('should show API error notification', () => {
      const error = { error: { message: 'API Error Message' } };
      const context = 'Test Context';
      
      const id = service.showApiError(error, context);
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('error');
      expect(notification?.title).toBe('Test Context Error');
      expect(notification?.message).toBe('API Error Message');
      expect(notification?.actions).toBeTruthy();
      expect(notification?.actions?.length).toBe(1);
      expect(notification?.actions?.[0].label).toBe('Retry');
    });

    it('should show API error with different error formats', () => {
      // Test with error.message
      const error1 = { message: 'Direct error message' };
      const id1 = service.showApiError(error1);
      expect(service.getById(id1)?.message).toBe('Direct error message');
      
      // Test with string error
      const error2 = 'String error message';
      const id2 = service.showApiError(error2);
      expect(service.getById(id2)?.message).toBe('String error message');
      
      // Test with unknown error format
      const error3 = { unknown: 'format' };
      const id3 = service.showApiError(error3);
      expect(service.getById(id3)?.message).toBe('An unexpected error occurred');
    });

    it('should show success notification with context', () => {
      const message = 'Operation completed';
      const context = 'User Profile';
      
      const id = service.showSuccess(message, context);
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('success');
      expect(notification?.title).toBe('User Profile Success');
      expect(notification?.message).toBe(message);
    });

    it('should show warning notification with context', () => {
      const message = 'Something to watch';
      const context = 'Data Import';
      
      const id = service.showWarning(message, context);
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('warning');
      expect(notification?.title).toBe('Data Import Warning');
      expect(notification?.message).toBe(message);
    });

    it('should show info notification with context', () => {
      const message = 'Information message';
      const context = 'System';
      
      const id = service.showInfo(message, context);
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('info');
      expect(notification?.title).toBe('System Info');
      expect(notification?.message).toBe(message);
    });

    it('should show loading notification', () => {
      const message = 'Loading data...';
      const title = 'Processing';
      
      const id = service.showLoading(message, title);
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('info');
      expect(notification?.title).toBe('Processing');
      expect(notification?.message).toBe(message);
      expect(notification?.autoClose).toBe(false);
      expect(notification?.dismissible).toBe(false);
    });

    it('should update loading notification to success', () => {
      const id = service.showLoading('Loading...');
      
      service.updateToSuccess(id, 'Operation completed successfully');
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('success');
      expect(notification?.message).toBe('Operation completed successfully');
      expect(notification?.autoClose).toBe(true);
      expect(notification?.dismissible).toBe(true);
    });

    it('should update loading notification to error', () => {
      const id = service.showLoading('Loading...');
      
      service.updateToError(id, 'Operation failed');
      
      const notification = service.getById(id);
      expect(notification?.type).toBe('error');
      expect(notification?.message).toBe('Operation failed');
      expect(notification?.autoClose).toBe(true);
      expect(notification?.dismissible).toBe(true);
    });
  });

  describe('Notification Limits', () => {
    beforeEach(() => {
      service.clear();
    });

    it('should limit notifications to maximum count', () => {
      // Add more than the default max (5) notifications
      for (let i = 0; i < 7; i++) {
        service.success(`Message ${i + 1}`);
      }
      
      expect(service.currentNotifications.length).toBe(5);
      expect(service.currentNotifications[0].message).toBe('Message 7'); // Most recent first
      expect(service.currentNotifications[4].message).toBe('Message 3'); // Oldest remaining
    });
  });

  describe('Observable Behavior', () => {
    it('should emit notifications through observable', (done) => {
      service.notifications.subscribe(notifications => {
        expect(notifications.length).toBe(1);
        expect(notifications[0].type).toBe('success');
        expect(notifications[0].message).toBe('Test message');
        done();
      });
      
      service.success('Test message');
    });

    it('should emit updated notifications when removing', (done) => {
      let callCount = 0;
      
      service.notifications.subscribe(notifications => {
        callCount++;
        
        if (callCount === 1) {
          expect(notifications.length).toBe(1);
          service.remove(notifications[0].id);
        } else if (callCount === 2) {
          expect(notifications.length).toBe(0);
          done();
        }
      });
      
      service.success('Test message');
    });
  });

  describe('Auto-close Functionality', () => {
    beforeEach(() => {
      jasmine.clock().install();
    });

    afterEach(() => {
      jasmine.clock().uninstall();
    });

    it('should auto-close notification after duration', () => {
      const id = service.success('Test message', undefined, { duration: 1000 });
      
      expect(service.exists(id)).toBe(true);
      
      jasmine.clock().tick(1000);
      
      expect(service.exists(id)).toBe(false);
    });

    it('should not auto-close when autoClose is false', () => {
      const id = service.show({
        type: 'info',
        message: 'Test message',
        autoClose: false,
        duration: 1000
      });
      
      expect(service.exists(id)).toBe(true);
      
      jasmine.clock().tick(1000);
      
      expect(service.exists(id)).toBe(true);
    });
  });
}); 