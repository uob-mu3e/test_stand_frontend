#pragma once

#include <iostream>

uint32_t evc;
uint32_t hits;
uint32_t ovrflw;
uint32_t rem;

void Set_EventCount(uint32_t ec) { evc = ec; }
void Set_Hits(uint32_t h) { hits = h; }
void Set_SubHeaderOvrflw(uint32_t ov) { ovrflw = ov; }
void Set_Reminders(uint32_t rm) { rem = rm; }

uint32_t Get_EventCount() { return evc; }
uint32_t Get_Hits() { return evc; }
uint32_t Get_SubHeaderOvrflw() { return evc; }
uint32_t Get_Reminders() { return evc; }
