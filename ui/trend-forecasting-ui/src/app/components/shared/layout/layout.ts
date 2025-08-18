import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { HeaderComponent } from '../header/header';
import { SidebarComponent, SidebarItem } from '../sidebar/sidebar';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner';

@Component({
  selector: 'app-layout',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatSidenavModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    HeaderComponent,
    SidebarComponent,
    LoadingSpinnerComponent
  ],
  templateUrl: './layout.html',
  styleUrls: ['./layout.scss']
})
export class LayoutComponent {
  @Input() sidebarItems: SidebarItem[] = [];
  @Input() activeRoute: string = '';
  @Input() loading: boolean = false;
  @Input() loadingMessage: string = 'Loading...';
  @Input() showSidebar: boolean = true;
  @Input() sidebarCollapsed: boolean = false;
  @Output() sidebarToggle = new EventEmitter<void>();
  @Output() sidebarItemClick = new EventEmitter<SidebarItem>();

  onSidebarToggle(): void {
    this.sidebarToggle.emit();
  }

  onSidebarItemClick(item: SidebarItem): void {
    this.sidebarItemClick.emit(item);
  }
}
