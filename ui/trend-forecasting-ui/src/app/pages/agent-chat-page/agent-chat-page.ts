import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AgentChatComponent } from '../../components/agent-chat/agent-chat';

@Component({
  selector: 'app-agent-chat-page',
  templateUrl: './agent-chat-page.html',
  styleUrls: ['./agent-chat-page.scss'],
  imports: [
    CommonModule,
    AgentChatComponent
  ],
  standalone: true
})
export class AgentChatPageComponent {
  // This is a page-level component that wraps the AgentChatComponent
}
