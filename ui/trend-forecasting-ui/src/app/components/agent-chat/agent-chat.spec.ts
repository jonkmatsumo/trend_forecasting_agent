import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentChat } from './agent-chat';

describe('AgentChat', () => {
  let component: AgentChat;
  let fixture: ComponentFixture<AgentChat>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentChat]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentChat);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
