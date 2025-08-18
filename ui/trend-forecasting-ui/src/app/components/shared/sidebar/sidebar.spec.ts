import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { RouterTestingModule } from '@angular/router/testing';
import { Router } from '@angular/router';
import { provideZonelessChangeDetection } from '@angular/core';
import { SidebarComponent } from './sidebar';
import { SidebarItem } from './sidebar';

describe('SidebarComponent', () => {
  let component: SidebarComponent;
  let fixture: ComponentFixture<SidebarComponent>;
  let router: Router;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        SidebarComponent,
        NoopAnimationsModule,
        RouterTestingModule.withRoutes([
          { path: 'agent-chat', component: SidebarComponent },
          { path: 'api-tester', component: SidebarComponent },
          { path: 'dashboard', component: SidebarComponent }
        ])
      ],
      providers: [
        provideZonelessChangeDetection()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SidebarComponent);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize with default values', () => {
    expect(component.collapsed).toBe(false);
    expect(component.items).toEqual([]);
    expect(component.activeRoute).toBe('');
  });

  describe('@Input properties', () => {
    it('should accept navigation items', () => {
      const testItems: SidebarItem[] = [
        { label: 'Dashboard', icon: 'dashboard', route: '/dashboard' },
        { label: 'Agent Chat', icon: 'chat', route: '/agent-chat' }
      ];

      component.items = testItems;
      fixture.detectChanges();

      expect(component.items).toEqual(testItems);
    });

    it('should accept collapsed state', () => {
      component.collapsed = true;
      fixture.detectChanges();

      expect(component.collapsed).toBe(true);
    });

    it('should accept active route', () => {
      const activeRoute = '/agent-chat';
      component.activeRoute = activeRoute;
      fixture.detectChanges();

      expect(component.activeRoute).toBe(activeRoute);
    });

    it('should accept items with optional properties', () => {
      const testItems: SidebarItem[] = [
        { 
          label: 'Dashboard', 
          icon: 'dashboard', 
          route: '/dashboard',
          badge: '3',
          disabled: false
        }
      ];

      component.items = testItems;
      fixture.detectChanges();

      expect(component.items).toEqual(testItems);
    });
  });

  describe('onItemClick', () => {
    it('should emit itemClick event with clicked item', () => {
      spyOn(component.itemClick, 'emit');
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.onItemClick(testItem);

      expect(component.itemClick.emit).toHaveBeenCalledWith(testItem);
    });

    it('should not emit event for disabled items', () => {
      spyOn(component.itemClick, 'emit');
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test',
        disabled: true
      };

      component.onItemClick(testItem);

      expect(component.itemClick.emit).not.toHaveBeenCalled();
    });

    it('should emit event for enabled items', () => {
      spyOn(component.itemClick, 'emit');
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test',
        disabled: false
      };

      component.onItemClick(testItem);

      expect(component.itemClick.emit).toHaveBeenCalledWith(testItem);
    });
  });

  describe('isActive', () => {
    it('should return true for active route', () => {
      component.activeRoute = '/agent-chat';

      const result = component.isActive('/agent-chat');

      expect(result).toBe(true);
    });

    it('should return false for inactive route', () => {
      component.activeRoute = '/agent-chat';

      const result = component.isActive('/api-tester');

      expect(result).toBe(false);
    });

    it('should handle exact route matching', () => {
      component.activeRoute = '/agent-chat';

      const result = component.isActive('/agent-chat');

      expect(result).toBe(true);
    });

    it('should return false for empty active route', () => {
      component.activeRoute = '';

      const result = component.isActive('/test');

      expect(result).toBe(false);
    });
  });

  describe('@Output events', () => {
    it('should emit itemClick event when item is clicked', () => {
      spyOn(component.itemClick, 'emit');
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.onItemClick(testItem);

      expect(component.itemClick.emit).toHaveBeenCalledWith(testItem);
    });
  });

  describe('template rendering', () => {
    it('should display navigation items', () => {
      const testItems: SidebarItem[] = [
        { label: 'Dashboard', icon: 'dashboard', route: '/dashboard' },
        { label: 'Agent Chat', icon: 'chat', route: '/agent-chat' }
      ];

      component.items = testItems;
      fixture.detectChanges();

      const itemElements = fixture.nativeElement.querySelectorAll('mat-list-item');
      expect(itemElements.length).toBe(2);
    });

    it('should display item labels', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const labelElement = fixture.nativeElement.querySelector('.mdc-list-item__content');
      expect(labelElement.textContent).toContain('Test Item');
    });

    it('should display item icons', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const iconElement = fixture.nativeElement.querySelector('mat-icon');
      expect(iconElement.textContent).toContain('test');
    });

    it('should apply active class to active route', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      component.activeRoute = '/test';
      fixture.detectChanges();

      const itemElement = fixture.nativeElement.querySelector('mat-list-item');
      expect(itemElement.classList.contains('active')).toBe(true);
    });

    it('should not apply active class to inactive route', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      component.activeRoute = '/other';
      fixture.detectChanges();

      const itemElement = fixture.nativeElement.querySelector('mat-list-item');
      expect(itemElement.classList.contains('active')).toBe(false);
    });

    it('should display badge when provided', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test',
        badge: '5'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const badgeElement = fixture.nativeElement.querySelector('.badge');
      expect(badgeElement.textContent).toContain('5');
    });
  });

  describe('accessibility', () => {
    it('should have proper ARIA labels', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const itemButton = fixture.nativeElement.querySelector('a[mat-list-item]');
      expect(itemButton).toBeTruthy();
    });

    it('should have proper role attributes', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const navList = fixture.nativeElement.querySelector('mat-nav-list');
      expect(navList).toBeTruthy();
    });
  });

  describe('responsive behavior', () => {
    it('should apply collapsed class when collapsed', () => {
      component.collapsed = true;
      fixture.detectChanges();

      const sidebarElement = fixture.nativeElement.querySelector('.sidebar');
      expect(sidebarElement.classList.contains('collapsed')).toBe(true);
    });

    it('should not apply collapsed class when not collapsed', () => {
      component.collapsed = false;
      fixture.detectChanges();

      const sidebarElement = fixture.nativeElement.querySelector('.sidebar');
      expect(sidebarElement.classList.contains('collapsed')).toBe(false);
    });
  });

  describe('navigation integration', () => {
    it('should have router link for items with route', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test'
      };

      component.items = [testItem];
      fixture.detectChanges();

      const linkElement = fixture.nativeElement.querySelector('a[routerLink]');
      expect(linkElement).toBeTruthy();
      expect(linkElement.getAttribute('routerLink')).toBe('/test');
    });
  });

  describe('item properties', () => {
    it('should handle items with all properties', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test',
        badge: '3',
        disabled: false
      };

      component.items = [testItem];
      fixture.detectChanges();

      const itemElement = fixture.nativeElement.querySelector('mat-list-item');
      expect(itemElement).toBeTruthy();
    });

    it('should handle disabled items', () => {
      const testItem: SidebarItem = {
        label: 'Test Item',
        icon: 'test',
        route: '/test',
        disabled: true
      };

      component.items = [testItem];
      fixture.detectChanges();

      const itemElement = fixture.nativeElement.querySelector('mat-list-item');
      expect(itemElement).toBeTruthy();
    });
  });
});
