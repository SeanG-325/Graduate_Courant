#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
char *FileName = NULL;
FILE *fp = NULL;
char buffer[10000] = { 0 };
int IOs[10000][2] = { 0 };
int IONum = 0;
int IONum1 = 0;
int EOFile = 0;
int Madd = 0;
int total_time = 0, tot_movement = 0, max_waittime = 0;
double avg_turnaround = 0, avg_waittime = 0;
class Event
{
public:
	int IO;
	int time;
	int track;
	int start_time;
	int sub_time;
	int end_time;
	Event()
	{
		IO = 0;
		time = 0;
		track = 0;
		start_time = 0;
		end_time = 0;
		sub_time = 0;
	}
};
Event Events[10000];
Event *E;
class IOschedule
{
public:
	virtual void add_queue(Event* e) = 0;
	virtual int AddMadd() = 0;
	virtual int EventNum() = 0;
	virtual Event* newEvent() = 0;
};
IOschedule *sched;
class FIFO : public IOschedule
{
public:
	queue <Event*> Q;
	void add_queue(Event* e)
	{
		Q.push(e);
	}
	int AddMadd()
	{
		if (Madd>E->track)
		{
			Madd--;
			return 1;
		}
		else if (Madd<E->track)
		{
			Madd++;
			return 1;
		}
		else return 0;
	}
	int EventNum()
	{
		return Q.size();
	}
	Event* newEvent()
	{
		Event* e = Q.front();
		Q.pop();
		return e;
	}
};
class SSTF : public IOschedule
{
public:
	vector <Event*> Q;
	void add_queue(Event* e)
	{
		Q.push_back(e);
	}
	int AddMadd()
	{
		if (Madd>E->track)
		{
			Madd--;
			return 1;
		}
		else if (Madd<E->track)
		{
			Madd++;
			return 1;
		}
		else return 0;
	}
	int EventNum()
	{
		return Q.size();
	}
	Event* newEvent()
	{
		Event* e = NULL;
		int m = 0x7fffffff, mindex = 0;
		for (unsigned int k = 0; k < Q.size(); k++)
		{
			if (abs(Q[k]->track - Madd) < m)
			{
				mindex = k;
				m = abs(Q[k]->track - Madd);
			}
		}
		e = Q[mindex];
		Q.erase(Q.begin() + mindex);
		return e;
	}
};
class LOOK : public IOschedule
{
public:
	vector <Event*> Q;
	int d, maxd, mind;
	LOOK()
	{
		d = 1;
		maxd = -1;
		mind = 0x7fffffff;
	}
	void add_queue(Event* e)
	{
		Q.push_back(e);
		if (maxd < e->track) maxd = e->track;
		if (mind > e->track) mind = e->track;
	}
	int AddMadd()
	{
		if (Madd < maxd && d == 1)
		{
			Madd++;
			return 1;
		}
		else if (Madd > mind && d == -1)
		{
			Madd--;
			return 1;
		}
		else return 0;
	}
	int EventNum()
	{
		return Q.size();
	}
	Event* newEvent()
	{
		Event* e = NULL;
		int m = 0x7fffffff, mindex = -1;
		int key = 0;
	aaa:
		key = 0;
		m = 0x7fffffff;
		mindex = -1;
		for (unsigned int k = 0; k < Q.size(); k++)
		{
			if (abs(Q[k]->track - Madd) < m && (d * (Q[k]->track - Madd)) >= 0)
			{
				mindex = k;
				m = abs(Q[k]->track - Madd);
				if (Q[k]->track - Madd == 0) key = 1;
			}
		}
		if (mindex != -1)
		{
			e = Q[mindex];
			if (d == 1)	maxd = e->track;
			else mind = e->track;
			Q.erase(Q.begin() + mindex);
			return e;
		}
		else
		{
			d = -d;
			goto aaa;
		}
	}
};
class CLOOK : public IOschedule
{
public:
	vector <Event*> Q;
	int d, maxd, mind;
	CLOOK()
	{
		d = 1;
		maxd = -1;
		mind = 0x7fffffff;
	}
	void add_queue(Event* e)
	{
		Q.push_back(e);
		if (maxd < e->track) maxd = e->track;
	}
	int AddMadd()
	{
		if (Madd < maxd && d == 1)
		{
			Madd++;
			return 1;
		}
		else if (Madd > mind && d == -1)
		{
			Madd--;
			return 1;
		}
		else return 0;
	}
	int EventNum()
	{
		return Q.size();
	}
	Event* newEvent()
	{
		Event* e = NULL;
		int m = 0x7fffffff, mindex = -1;
		int key = 0;
		key = 0;
		m = 0x7fffffff;
		mindex = -1;
		d = 1;
		for (unsigned int k = 0; k < Q.size(); k++)
		{
			if (abs(Q[k]->track - Madd) < m && (d * (Q[k]->track - Madd)) >= 0)
			{
				mindex = k;
				m = abs(Q[k]->track - Madd);
			}
		}
		if (mindex != -1)
		{
			e = Q[mindex];
			maxd = e->track;
			Q.erase(Q.begin() + mindex);
			return e;
		}
		else
		{
			d = -1;
			m = -1;
			for (unsigned int k = 0; k < Q.size(); k++)
			{
				if (abs(Q[k]->track - Madd) > m && (d * (Q[k]->track - Madd)) >= 0)
				{
					mindex = k;
					m = abs(Q[k]->track - Madd);
				}
			}
			e = Q[mindex];
			mind = e->track;
			Q.erase(Q.begin() + mindex);
			return e;
		}
	}
};
class FLOOK : public IOschedule
{
public:
	vector <Event*> Q1;
	vector <Event*> Q2;
	vector <Event*> *aQ;
	vector <Event*> *uQ;
	int d, maxd, mind;
	FLOOK()
	{
		d = 1;
		maxd = -1;
		mind = 0x7fffffff;
		aQ = &Q1;
		uQ = &Q2;
	}
	void add_queue(Event* e)
	{
		uQ->push_back(e);
		if (maxd < e->track) maxd = e->track;
		if (mind > e->track) mind = e->track;
	}
	int AddMadd()
	{
		if (Madd < maxd && d == 1)
		{
			Madd++;
			return 1;
		}
		else if (Madd > mind && d == -1)
		{
			Madd--;
			return 1;
		}
		else return 0;
	}
	int EventNum()
	{
		return aQ->size() + uQ->size();
	}
	Event* newEvent()
	{
		Event* e = NULL;
		int m = 0x7fffffff, mindex = -1;
		int key = 0;
		if (aQ->size() == 0)
		{
			vector <Event*> *q;
			q = aQ;
			aQ = uQ;
			uQ = q;
		}
	aaa:
		key = 0;
		m = 0x7fffffff;
		mindex = -1;
		for (unsigned int k = 0; k < aQ->size(); k++)
		{
			if (abs((*aQ)[k]->track - Madd) < m && (d * ((*aQ)[k]->track - Madd)) >= 0)
			{
				mindex = k;
				m = abs((*aQ)[k]->track - Madd);
				if ((*aQ)[k]->track - Madd == 0) key = 1;
			}
		}
		if (mindex != -1)
		{
			e = (*aQ)[mindex];
			if (d == 1)	maxd = e->track;
			else mind = e->track;
			aQ->erase(aQ->begin() + mindex);
			return e;
		}
		else
		{
			d = -d;
			goto aaa;
		}
	}
};
void Createbuffer()
{
	char* i;
	if (fp == NULL)
	{
		fp = fopen(FileName, "r");
	}
	while (1)
	{
		i = fgets(buffer, 10000, fp);
		if (i == NULL)
		{
			EOFile = 1;
			buffer[0] = 0;
			return;
		}
		if (buffer[0] == '#' || buffer[0] == '\n') continue;
		return;
	}
}
void CreateIO()
{
	int temp = 0;
	while (1)
	{
		Createbuffer();
		if (EOFile == 1) break;
		else
		{
			sscanf(buffer, "%d %d", &IOs[temp][0], &IOs[temp][1]);
			Events[temp].IO = temp;
			Events[temp].time = IOs[temp][0];
			Events[temp].track = IOs[temp][1];
			temp++;
		}
	}
	IONum = temp;
	IONum1 = IONum;
	fclose(fp);
}
void Simulation()
{
	int i = 0, a = 0, m = 0;
	while (1)
	{
		if (Events[a].time == i)
		{
			sched->add_queue(&(Events[a++]));
		}
		if (E != NULL&&Madd == E->track)
		{
			E->end_time = i;
			E = NULL;
			total_time = i;
			IONum1--;
		}
		if (E != NULL&&Madd != E->track)
		{
			m = sched->AddMadd();
			if (m != 0) tot_movement++;
		}
		if (E == NULL&&sched->EventNum() != 0)
		{
			E = sched->newEvent();
			E->start_time = i;
			if (max_waittime < E->start_time - E->time) max_waittime = E->start_time - E->time;
			continue;
		}
		if (E == NULL && sched->EventNum() == 0 && IONum1 == 0)
		{
			break;
		}
		i++;
	}
	for (int k = 0; k<IONum; k++)
	{
		avg_waittime += (double)(Events[k].start_time - Events[k].time);
		avg_turnaround += (double)(Events[k].end_time - Events[k].time);
	}
	avg_waittime /= (double)IONum;
	avg_turnaround /= (double)IONum;
	for (int k = 0; k<IONum; k++)
	{
		printf("%5d: %5d %5d %5d\n", k, Events[k].time, Events[k].start_time, Events[k].end_time);
	}
	printf("SUM: %d %d %.2lf %.2lf %d\n", total_time, tot_movement, avg_turnaround, avg_waittime, max_waittime);
}
int main(int argc, char **argv)
{
	char *s_value = NULL;
	int inst;
	while ((inst = getopt(argc, argv, "s:")) != -1)
	{
		switch (inst)
		{
		case 's':
			s_value = optarg;
			switch (s_value[0])
			{
			case 'i':
				sched = new FIFO;
				break;
			case 'j':
				sched = new SSTF;
				break;
			case 's':
				sched = new LOOK;
				break;
			case 'c':
				sched = new CLOOK;
				break;
			case 'f':
				sched = new FLOOK;
				break;
			}
			break;
		case '?':
			break;
		default:
			break;
		}
	}
	FileName = argv[optind];
	CreateIO();
	Simulation();
}
