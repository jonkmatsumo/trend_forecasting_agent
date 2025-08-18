import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface SkeletonConfig {
  type: 'text' | 'circle' | 'rectangle' | 'card' | 'list-item' | 'table-row';
  width?: string;
  height?: string;
  borderRadius?: string;
  lines?: number;
  lineHeight?: string;
  spacing?: string;
}

@Component({
  selector: 'app-skeleton-loader',
  templateUrl: './skeleton-loader.html',
  styleUrls: ['./skeleton-loader.scss'],
  imports: [CommonModule],
  standalone: true
})
export class SkeletonLoaderComponent implements OnInit {
  @Input() config: SkeletonConfig = { type: 'text' };
  @Input() count: number = 1;
  @Input() animated: boolean = true;
  @Input() className: string = '';

  skeletonItems: SkeletonConfig[] = [];

  ngOnInit(): void {
    this.generateSkeletonItems();
  }

  private generateSkeletonItems(): void {
    this.skeletonItems = Array(this.count).fill(null).map(() => ({
      ...this.config,
      width: this.config.width || this.getDefaultWidth(),
      height: this.config.height || this.getDefaultHeight(),
      borderRadius: this.config.borderRadius || this.getDefaultBorderRadius(),
      lines: this.config.lines || this.getDefaultLines(),
      lineHeight: this.config.lineHeight || this.getDefaultLineHeight(),
      spacing: this.config.spacing || this.getDefaultSpacing()
    }));
  }

  private getDefaultWidth(): string {
    switch (this.config.type) {
      case 'circle':
        return '40px';
      case 'rectangle':
        return '100%';
      case 'card':
        return '300px';
      case 'list-item':
        return '100%';
      case 'table-row':
        return '100%';
      default:
        return '100%';
    }
  }

  private getDefaultHeight(): string {
    switch (this.config.type) {
      case 'circle':
        return '40px';
      case 'rectangle':
        return '20px';
      case 'card':
        return '200px';
      case 'list-item':
        return '60px';
      case 'table-row':
        return '50px';
      default:
        return '16px';
    }
  }

  private getDefaultBorderRadius(): string {
    switch (this.config.type) {
      case 'circle':
        return '50%';
      case 'card':
        return '8px';
      case 'rectangle':
        return '4px';
      default:
        return '2px';
    }
  }

  private getDefaultLines(): number {
    switch (this.config.type) {
      case 'text':
        return 3;
      case 'card':
        return 4;
      case 'list-item':
        return 2;
      case 'table-row':
        return 1;
      default:
        return 1;
    }
  }

  private getDefaultLineHeight(): string {
    switch (this.config.type) {
      case 'text':
        return '16px';
      case 'card':
        return '14px';
      case 'list-item':
        return '16px';
      case 'table-row':
        return '16px';
      default:
        return '16px';
    }
  }

  private getDefaultSpacing(): string {
    switch (this.config.type) {
      case 'text':
        return '8px';
      case 'card':
        return '12px';
      case 'list-item':
        return '8px';
      case 'table-row':
        return '4px';
      default:
        return '8px';
    }
  }

  getSkeletonClass(): string {
    const baseClass = 'skeleton-loader';
    const typeClass = `skeleton-${this.config.type}`;
    const animationClass = this.animated ? 'skeleton-animated' : '';
    return `${baseClass} ${typeClass} ${animationClass} ${this.className}`.trim();
  }

  getLineWidths(): string[] {
    if (this.config.type === 'text' || this.config.type === 'card') {
      return Array(this.config.lines || 1).fill(null).map((_, index) => {
        // Vary line widths for more realistic appearance
        const widths = ['100%', '85%', '70%', '90%', '60%'];
        return widths[index % widths.length];
      });
    }
    return ['100%'];
  }

  trackByIndex(index: number): number {
    return index;
  }
} 