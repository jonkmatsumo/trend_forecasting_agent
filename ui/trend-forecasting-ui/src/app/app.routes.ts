import { Routes } from '@angular/router';
import { AgentChatPageComponent } from './pages/agent-chat-page/agent-chat-page';
import { ApiTesterPageComponent } from './pages/api-tester-page/api-tester-page';

export const routes: Routes = [
  { path: '', redirectTo: '/agent-chat', pathMatch: 'full' },
  { path: 'agent-chat', component: AgentChatPageComponent },
  { path: 'api-tester', component: ApiTesterPageComponent },
  { path: '**', redirectTo: '/agent-chat' }
];
