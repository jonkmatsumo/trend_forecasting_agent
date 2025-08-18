import { Routes } from '@angular/router';
import { AgentChatPageComponent } from './pages/agent-chat-page/agent-chat-page';
import { ApiTesterPageComponent } from './pages/api-tester-page/api-tester-page';
import { StylingDemoPageComponent } from './pages/styling-demo-page/styling-demo-page';
import { ErrorHandlingDemoPageComponent } from './pages/error-handling-demo-page/error-handling-demo-page';

export const routes: Routes = [
  { path: '', redirectTo: '/agent-chat', pathMatch: 'full' },
  { path: 'agent-chat', component: AgentChatPageComponent },
  { path: 'api-tester', component: ApiTesterPageComponent },
  { path: 'styling-demo', component: StylingDemoPageComponent },
  { path: 'error-handling-demo', component: ErrorHandlingDemoPageComponent },
  { path: '**', redirectTo: '/agent-chat' }
];
