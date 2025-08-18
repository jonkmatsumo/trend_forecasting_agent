import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';

export interface SidebarItem {
  label: string;
  icon: string;
  route: string;
  badge?: string;
  disabled?: boolean;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatListModule,
    MatIconModule,
    MatButtonModule,
    MatDividerModule,
    MatTooltipModule
  ],
  templateUrl: './sidebar.html',
  styleUrls: ['./sidebar.scss']
})
export class SidebarComponent {
  @Input() items: SidebarItem[] = [];
  @Input() collapsed: boolean = false;
  @Input() activeRoute: string = '';
  @Output() itemClick = new EventEmitter<SidebarItem>();

  onItemClick(item: SidebarItem): void {
    if (!item.disabled) {
      this.itemClick.emit(item);
    }
  }

  isActive(route: string): boolean {
    return this.activeRoute === route;
  }
}
