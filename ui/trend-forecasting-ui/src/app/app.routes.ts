import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: '/agent-chat', pathMatch: 'full' },
  { 
    path: 'agent-chat', 
    loadComponent: () => import('./pages/agent-chat-page/agent-chat-page').then(m => m.AgentChatPageComponent)
  },
  { 
    path: 'api-tester', 
    loadComponent: () => import('./pages/api-tester-page/api-tester-page').then(m => m.ApiTesterPageComponent)
  },
  { 
    path: 'styling-demo', 
    loadComponent: () => import('./pages/styling-demo-page/styling-demo-page').then(m => m.StylingDemoPageComponent)
  },
  { 
    path: 'error-handling-demo', 
    loadComponent: () => import('./pages/error-handling-demo-page/error-handling-demo-page').then(m => m.ErrorHandlingDemoPageComponent)
  },
  { 
    path: 'performance', 
    loadComponent: () => import('./pages/performance-page/performance-page').then(m => m.PerformancePageComponent)
  },
  { path: '**', redirectTo: '/agent-chat' }
];
