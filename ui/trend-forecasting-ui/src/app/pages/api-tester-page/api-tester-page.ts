import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiTesterComponent } from '../../components/api-tester/api-tester';

@Component({
  selector: 'app-api-tester-page',
  templateUrl: './api-tester-page.html',
  styleUrls: ['./api-tester-page.scss'],
  imports: [
    CommonModule,
    ApiTesterComponent
  ],
  standalone: true
})
export class ApiTesterPageComponent {
  // This is a page-level component that wraps the ApiTesterComponent
}
