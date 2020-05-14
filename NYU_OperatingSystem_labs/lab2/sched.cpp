#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <queue>
#include <vector>
#include <stack>
#include <algorithm>
using namespace std;
int Details = 0;
char *FileName = NULL, *FileName2 = NULL;
int *Numbers = NULL, Num = 0;
int quantum, maxprio = 4;
enum Models { _FCFS, _LCFS, _SRTF, _RR, _PRIO, _PREPRIO };
enum States { CREATE, READY, RUNNING, BLOCKED, PREEMPTED, COMPLETED };
enum TRANS { TRANS_TO_READY, TRANS_TO_RUN, TRANS_TO_BLOCK, TRANS_TO_PREEMPTED, TRANS_TO_COMPLETED};
int ProcessNum = 0;
int CURRENT_TIME = 0;
bool CALL_SCHEDULER = false;
int ofs = 0;
int CompletedTime = 0;
int CPUTime = 0;
int IOTime = 0;
int TurnaroundTime = 0;
int CpuWaitingTime = 0;
int i1 = 0, i2 = 0;
class Process
{
public:
	int PID;
	int AT, TC, CB, IO;
	int state_ts;
	int cpu_burst;
	int io_burst;
	int s_prio, d_prio;
	int rem_time;
	int last_running;
	int last_io;
	int FT, TT, IT, CW;
	Process()
	{
		AT = -1;
		TC = 0;
		CB = 0;
		IO = 0;
		state_ts = 0;
		rem_time = TC;
	}
};
class Event
{
public:
	int TimeStamp;
	Process* process;
	States OldState, NewState;
	TRANS T;
};
queue<Process*> ProcessesQueue;
vector<Event*> Events;
Process Processes[10000];
Process *CURRENT_RUNNING_PROCESS = NULL;
void CreateInitialEvents()
{
	for (int i = 0; i<ProcessNum; i++)
	{
		Event *temp = NULL;
		temp = (Event*)malloc(sizeof(Event));
		temp->NewState = CREATE;
		temp->OldState = CREATE;
		temp->process = &Processes[i];
		temp->TimeStamp = temp->process->AT;
		temp->T = TRANS_TO_READY;
		Events.push_back(temp);
	}
}
int Cmp2(Process *p1, Process *p2)
{
	return p1->rem_time < p2->rem_time;
}
int Cmp(Event* e1, Event* e2)
{
	return e1->TimeStamp < e2->TimeStamp;
}
Event* get_Event()
{
	Event* temp = NULL;
	if (Events.size())
	{
		stable_sort(Events.begin(), Events.end(), Cmp);
		temp = Events.front();
		Events.erase(Events.begin());
	}
	return temp;
}
void Add_Event(Event* E)
{
	Events.push_back(E);
	stable_sort(Events.begin(), Events.end(), Cmp);
}
class Scheduler
{
public:
	Models Sche;
	Process* temp;
	virtual void add_process(Process *p) = 0;
	virtual Process* get_next_process() = 0;
	virtual void test_preempt(Process *p, int curtime){}
};
class FCFS : public Scheduler
{
public:
	queue <Process*> RUN_QUEUE;
	void add_process(Process *p)
	{
		if (p->d_prio < 0) p->d_prio = p->s_prio - 1;
		RUN_QUEUE.push(p);
	}
	Process* get_next_process()
	{
		if (RUN_QUEUE.size())
		{
			temp = RUN_QUEUE.front();
			RUN_QUEUE.pop();
			return temp;
		}
		else return NULL;
	}
	FCFS()
	{
		Sche = _FCFS;
	}
};
class LCFS : public Scheduler
{
public:
	stack <Process*> RUN_QUEUE;
	void add_process(Process *p)
	{
		if (p->d_prio < 0) p->d_prio = p->s_prio - 1;
		RUN_QUEUE.push(p);
	}
	Process* get_next_process()
	{
		if (RUN_QUEUE.size())
		{
			temp = RUN_QUEUE.top();
			RUN_QUEUE.pop();
			return temp;
		}
		else return NULL;
	}
	LCFS()
	{
		Sche = _LCFS;
	}
};
class SRTF : public Scheduler
{
public:
	vector <Process*> RUN_QUEUE;
	void add_process(Process *p)
	{
		if (p->d_prio < 0) p->d_prio = p->s_prio - 1;
		RUN_QUEUE.push_back(p);
		stable_sort(RUN_QUEUE.begin(), RUN_QUEUE.end(), Cmp2);
	}
	Process* get_next_process()
	{
		if (RUN_QUEUE.size())
		{
			temp = RUN_QUEUE.front();
			RUN_QUEUE.erase(RUN_QUEUE.begin());
			return temp;
		}
		else return NULL;
	}
	SRTF()
	{
		Sche = _SRTF;
	}
};
class RR : public Scheduler
{
public:
	queue <Process*> RUN_QUEUE;
	void add_process(Process *p)
	{
		if (p->d_prio < 0) p->d_prio = p->s_prio - 1;
		RUN_QUEUE.push(p);
	}
	Process* get_next_process()
	{
		if (RUN_QUEUE.size())
		{
			temp = RUN_QUEUE.front();
			RUN_QUEUE.pop();
			return temp;
		}
		else return NULL;
	}
	RR()
	{
		Sche = _RR;
	}
};
class PRIO : public Scheduler
{
public:
	vector <queue <Process*> *> *activeQ, *expiredQ, *temp2;
	PRIO()
	{
		Sche = _PRIO;
		activeQ = new vector <queue <Process*> *>;
		expiredQ = new vector <queue <Process*> *>;
		for (int i = 0; i < maxprio; i++)
		{
			queue <Process*> *temp1 = new queue <Process*>;
			activeQ->push_back(temp1);
			temp1 = new queue <Process*>;
			expiredQ->push_back(temp1);
		}
	}
	void add_process(Process *p)
	{
		if (p->d_prio >= 0) (*activeQ)[p->d_prio]->push(p);
		else
		{
			p->d_prio = p->s_prio - 1;
			expiredQ->at(p->d_prio)->push(p);
		}
	}
	Process* get_next_process()
	{
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				activeQ->at(i)->pop();
				return temp;
			}
		}
		temp2 = activeQ;
		activeQ = expiredQ;
		expiredQ = temp2;
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				activeQ->at(i)->pop();
				return temp;
			}
		}
		return NULL;
	}
};
class PREPRIO : public Scheduler
{
public:
	vector <queue <Process*> *> *activeQ, *expiredQ, *temp2;
	PREPRIO()
	{
		Sche = _PREPRIO;
		activeQ = new vector <queue <Process*> *>;
		expiredQ = new vector <queue <Process*> *>;
		for (int i = 0; i < maxprio; i++)
		{
			queue <Process*> *temp1 = new queue <Process*>;
			activeQ->push_back(temp1);
			temp1 = new queue <Process*>;
			expiredQ->push_back(temp1);
		}
	}
	void add_process(Process *p)
	{
		if (p->d_prio >= 0) (*activeQ)[p->d_prio]->push(p);
		else
		{
			p->d_prio = p->s_prio - 1;
			expiredQ->at(p->d_prio)->push(p);
		}
	}
	Process* get_next_process()
	{
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				activeQ->at(i)->pop();
				return temp;
			}
		}
		temp2 = activeQ;
		activeQ = expiredQ;
		expiredQ = temp2;
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				activeQ->at(i)->pop();
				return temp;
			}
		}
		return NULL;
	}
	Process* see_next_process()
	{
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				return temp;
			}
		}
		temp2 = activeQ;
		activeQ = expiredQ;
		expiredQ = temp2;
		for (int i = maxprio - 1; i >= 0; i--)
		{
			if (activeQ->at(i)->size())
			{
				temp = activeQ->at(i)->front();
				return temp;
			}
		}
		return NULL;
	}
	void test_preempt(Process *p, int curtime)
	{
		Process* temp3 = see_next_process();
		if (temp3->d_prio > p->d_prio)
		{
			for (int i = 0; i < Events.size(); i++)
			{
				if (Events[i]->process == p && Events[i]->TimeStamp == curtime)
				{
					return;
				}
			}
			for (vector<Event*>::iterator it = Events.begin(); it != Events.end();)
			{
				if ((*it)->process == p)
				{
					Events.erase(it);
					it = Events.begin();
				}
				else it++;
			}
			Event* eNew = (Event*)malloc(sizeof(Event));
			eNew->TimeStamp = curtime;
			eNew->process = p;
			eNew->process->cpu_burst += eNew->process->last_running - (curtime - eNew->process->state_ts);
			eNew->process->rem_time += eNew->process->last_running - (curtime - eNew->process->state_ts);
			eNew->process->last_running = curtime - eNew->process->state_ts;
			eNew->OldState = RUNNING;
			eNew->NewState = PREEMPTED;
			eNew->T = TRANS_TO_PREEMPTED;
			Add_Event(eNew);
		}
	}
};
int myrandom(int burst) 
{
	return 1 + (Numbers[(ofs++) % Num] % burst);
}
int get_next_event_time()
{
	if (Events.size() > 0) return Events.front()->TimeStamp;
	else return -1;
}
void Simulation(Scheduler* S)
{
	int preemption = 0;
	Event *evt, *eNew;
	int timeInPrevState;
	int key = 0, key2 = 0;
	while (evt = get_Event())
	{
		Process *proc = evt->process;
		CURRENT_TIME = evt->TimeStamp;
		timeInPrevState = CURRENT_TIME - proc->state_ts;
		switch (evt->T)
		{
		case TRANS_TO_READY:
			if (Details)
			{
				printf("%d %d %d: ", CURRENT_TIME, proc->PID, CURRENT_TIME - proc->state_ts);
				if (evt->OldState == CREATE) printf("CREATE -> READY\n");
				else if(evt->OldState == BLOCKED) printf("BLOCKED -> READY\n");
				else printf("RUN -> READY\n");
			}
			proc->state_ts = CURRENT_TIME;
			S->add_process(proc);
			if (S->Sche == _PREPRIO && CURRENT_RUNNING_PROCESS != NULL)	S->test_preempt(CURRENT_RUNNING_PROCESS, CURRENT_TIME);
			if (!CURRENT_RUNNING_PROCESS) CALL_SCHEDULER = 1;
			proc->IT += timeInPrevState;
			break;
		case TRANS_TO_RUN:
			if (Details)
			{
				printf("%d %d %d: ", CURRENT_TIME, proc->PID, CURRENT_TIME - proc->state_ts);
				printf("READY -> RUN\n");
			}
			if (S->Sche == _FCFS || S->Sche == _LCFS || S->Sche == _SRTF) quantum = 10000;
			eNew = (Event*)malloc(sizeof(Event));
			eNew->TimeStamp = evt->TimeStamp;
			eNew->process = evt->process;
			if (eNew->process->cpu_burst == 0) key = myrandom(eNew->process->CB);
			else key = eNew->process->cpu_burst;
			if (key > quantum)
			{
				key2 = quantum;
				preemption = 1;
			}
			else key2 = key;
			eNew->process->state_ts = eNew->TimeStamp;
			eNew->process->last_running = eNew->process->rem_time >= key2 ? key2 : eNew->process->rem_time;//get running time for this time
			eNew->process->rem_time -= eNew->process->last_running;//update rem_time
			eNew->TimeStamp += eNew->process->last_running;
			if (eNew->process->rem_time == 0)
			{
				eNew->OldState = RUNNING;
				eNew->NewState = COMPLETED;
				eNew->T = TRANS_TO_COMPLETED;
				Add_Event(eNew);
			}
			else if (preemption)
			{
				preemption = 0;
				eNew->process->cpu_burst = key - quantum;
				eNew->OldState = RUNNING;
				eNew->NewState = PREEMPTED;
				eNew->T = TRANS_TO_PREEMPTED;
				Add_Event(eNew);
			}
			else
			{
				eNew->process->cpu_burst = 0;
				eNew->OldState = RUNNING;
				eNew->NewState = BLOCKED;
				eNew->T = TRANS_TO_BLOCK;
				Add_Event(eNew);
			}
			preemption = 0;
			break;
		case TRANS_TO_BLOCK:
			eNew = (Event*)malloc(sizeof(Event));
			eNew->TimeStamp = evt->TimeStamp;
			eNew->process = evt->process;
			eNew->process->state_ts = eNew->TimeStamp;
			eNew->process->last_io = myrandom(eNew->process->IO);
			eNew->TimeStamp += eNew->process->last_io;
			if (i2 <= CURRENT_TIME)
			{
				IOTime += (i2 - i1);
				i1 = CURRENT_TIME;
				i2 = CURRENT_TIME + eNew->process->last_io;
			}
			else i2 = (i2 >= CURRENT_TIME + eNew->process->last_io ? i2 : CURRENT_TIME + eNew->process->last_io);
			if (Details)
			{
				printf("%d %d %d: ", CURRENT_TIME, proc->PID, CURRENT_TIME - proc->state_ts);
				printf("RUN -> BLOCKED ib=%d  rem=%d\n", proc->last_io, proc->rem_time);
			}
			proc->d_prio = proc->s_prio - 1;
			eNew->OldState = BLOCKED;
			eNew->NewState = READY;
			eNew->T = TRANS_TO_READY;
			Add_Event(eNew);
			CALL_SCHEDULER = 1;
			CPUTime += timeInPrevState;
			break;
		case TRANS_TO_PREEMPTED:
			if (Details)
			{
				printf("%d %d %d: ", CURRENT_TIME, proc->PID, CURRENT_TIME - proc->state_ts);
				printf("RUN -> READY\n");
			}
			CALL_SCHEDULER = 1;
			CPUTime += timeInPrevState;
			proc->state_ts = CURRENT_TIME;
			proc->d_prio--;
			S->add_process(evt->process);
			break;
		case TRANS_TO_COMPLETED:

			if (Details)
			{
				printf("%d %d %d: ", CURRENT_TIME, proc->PID, CURRENT_TIME - proc->state_ts);
				printf("RUN -> COMPLETED\n");
			}
			proc->FT = CURRENT_TIME;
			proc->TT = CURRENT_TIME - proc->AT;
			CPUTime += timeInPrevState;
			TurnaroundTime += (CURRENT_TIME - proc->AT);
			CompletedTime = CURRENT_TIME;//update complete time
			CALL_SCHEDULER = 1;
			break;
		}
		free(evt);
		evt = NULL;
		if (CALL_SCHEDULER)
		{
			if (CURRENT_RUNNING_PROCESS == proc) //block/preempted event come and free cpu 
			{
				CURRENT_RUNNING_PROCESS = NULL;
			}
			if (get_next_event_time() == CURRENT_TIME) continue;
			CALL_SCHEDULER = 0;
			if (CURRENT_RUNNING_PROCESS == NULL)
			{
				CURRENT_RUNNING_PROCESS = S->get_next_process();
				if (CURRENT_RUNNING_PROCESS == NULL) continue;
				eNew = (Event*)malloc(sizeof(Event));
				eNew->TimeStamp = CURRENT_TIME;
				eNew->process = CURRENT_RUNNING_PROCESS;
				eNew->process->CW += CURRENT_TIME - eNew->process->state_ts;
				CpuWaitingTime += CURRENT_TIME - eNew->process->state_ts;
				eNew->process->state_ts = eNew->TimeStamp;
				eNew->T = TRANS_TO_RUN;
				eNew->OldState = READY;
				eNew->NewState = RUNNING;
				Add_Event(eNew);
			}
		}
	}
	IOTime += i2 - i1;
}
void CreateProcesses()
{
	int i = 0;
	char buffer[100] = { 0 };
	FILE *fp = NULL;
	fp = fopen(FileName, "r");
	while (fgets(buffer, 100, fp))
	{
		sscanf(buffer, "%d %d %d %d", &Processes[i].AT, &Processes[i].TC, &Processes[i].CB, &Processes[i].IO);
		Processes[i].PID = i;
		Processes[i].rem_time = Processes[i].TC;
		Processes[i].state_ts = Processes[i].AT;
		Processes[i].s_prio = myrandom(maxprio);
		Processes[i].d_prio = Processes[i].s_prio - 1;
		Processes[i].cpu_burst = 0;
		i++;
	}
	fclose(fp);
	ProcessNum = i;
}
void CreaterandomNumbers()
{
	FILE *fp = NULL;
	fp = fopen(FileName2, "r");
	fscanf(fp, "%d", &Num);
	Numbers = (int*)malloc(sizeof(int)*Num);
	for (int i = 0; i<Num; i++)	fscanf(fp, "%d", Numbers + i);
	fclose(fp);
}
int main(int argc, char **argv)
{
	char *s_value = NULL;
	int ins;
	Scheduler *schem;
	while ((ins = getopt(argc, argv, "etvs:")) != -1)
	{
		switch (ins)
		{
		case 'v':
			Details = 1;
			break;
		case 's':
			s_value = optarg;
			switch (s_value[0])
			{
			case 'F':
				schem = new FCFS;
				printf("FCFS\n");
				break;
			case 'S':
				schem = new SRTF;
				printf("SRTF\n");
				break;
			case 'L':
				schem = new LCFS;
				printf("LCFS\n");
				break;
			case 'R':
				s_value++;
				sscanf(s_value, "%d:%d", &quantum, &maxprio);
				schem = new RR;
				printf("RR %d\n",quantum);
				break;
			case 'P':
				s_value++;
				sscanf(s_value, "%d:%d", &quantum, &maxprio);
				schem = new PRIO;
				printf("PRIO %d\n", quantum);
				break;
			case 'E':
				s_value++;
				sscanf(s_value, "%d:%d", &quantum, &maxprio);
				schem = new PREPRIO;
				printf("PREPRIO %d\n", quantum);
				break;
			}
		case '?':
			break;
		default:
			break;
		}
	}
	FileName = argv[optind];
	FileName2 = argv[optind + 1];
	CreaterandomNumbers();
	CreateProcesses();
	CreateInitialEvents();
	Simulation(schem);
	for (int i = 0; i < ProcessNum; i++)
	{
		printf("%04d: %4d %4d %4d %4d %1d", Processes[i].PID, Processes[i].AT, Processes[i].TC, Processes[i].CB, Processes[i].IO, Processes[i].s_prio);
		printf(" | %5d %5d %5d %5d\n", Processes[i].FT, Processes[i].TT, Processes[i].IT, Processes[i].CW);
	}
	printf("SUM: %d %.2lf %.2lf %.2lf %.2lf %.3lf\n", CompletedTime, double(CPUTime * 100) / double(CompletedTime), double(IOTime * 100) / double(CompletedTime), double(TurnaroundTime) / double(ProcessNum), double(CpuWaitingTime) / double(ProcessNum), double(ProcessNum) / double(CompletedTime) * double(100));
}
