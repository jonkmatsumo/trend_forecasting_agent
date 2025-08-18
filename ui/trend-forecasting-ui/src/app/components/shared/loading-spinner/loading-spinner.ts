import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

@Component({
  selector: 'app-loading-spinner',
  standalone: true,
  imports: [
    CommonModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './loading-spinner.html',
  styleUrls: ['./loading-spinner.scss']
})
export class LoadingSpinnerComponent {
  @Input() size: 'small' | 'medium' | 'large' = 'medium';
  @Input() overlay: boolean = false;
  @Input() message: string = 'Loading...';
  @Input() color: 'primary' | 'accent' | 'warn' = 'primary';

  get spinnerSize(): number {
    switch (this.size) {
      case 'small': return 24;
      case 'large': return 64;
      default: return 48;
    }
  }

  get spinnerStrokeWidth(): number {
    switch (this.size) {
      case 'small': return 2;
      case 'large': return 4;
      default: return 3;
    }
  }
}
