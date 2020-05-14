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
#include <string.h>
#define PTENUM 64
#define MAX_FRAMES 128
#define COST_INOUT 3000
#define COST_MAPUNMAP 400
#define COST_FINFOUT 2500
#define COST_Z 150
#define COST_SEGV 240
#define COST_SEGPROT 300
#define COST_RW 1
#define COST_CTW 121
#define COST_E 175
#define _ESC_NRU 50
#define WS 49
using namespace std;
class VMA1
{
public:
	int start_vpage;
	int end_vpage;
	int write_protected;
	int file_mapped;
};
class pte_t
{
public:
	unsigned int PRESENT_VALID : 1;
	unsigned int frames : 7;
	unsigned int WRITE_PROTECT : 1;
	unsigned int MODIFIED : 1;
	unsigned int REFERENCED : 1;
	unsigned int PAGEDOUT : 1;
	unsigned int used : 1;
	unsigned int set : 1;
	unsigned int sw : 1;
	unsigned int : 17;
	pte_t()
	{
		PRESENT_VALID = 0;
		frames = 0;
		used = 0;
		sw = 0;
		set = 0;
		PAGEDOUT = 0;
		REFERENCED = 0;
		MODIFIED = 0;
		WRITE_PROTECT = 0;
		frames = 0;
		PRESENT_VALID = 0;
	}
};
class pstats
{
public:
	long unsigned int unmaps;
	long unsigned int maps;
	long unsigned int ins;
	long unsigned int outs;
	long unsigned int fins;
	long unsigned int fouts;
	long unsigned int zero;
	long unsigned int segv;
	long unsigned int segprot;
	pstats()
	{
		unmaps = 0;
		maps = 0;
		ins = 0;
		outs = 0;
		fins = 0;
		fouts = 0;
		zero = 0;
		segv = 0;
		segprot = 0;
	}
};
class Process
{
public:
	int pid;
	int VMANum;
	VMA1 *VMA;
	pte_t page_table[PTENUM];
	pstats stats;
};
class frame_t
{
public:
	int fid;
	int mappedProcess;
	int mappedvpage;
	int mapped;
	int sw;
	unsigned int age;
	unsigned int lastusedtime;
	frame_t()
	{
		fid = -1;
		mappedProcess = -1;
		mappedvpage = -1;
		mapped = 0;
		sw = 0;
		age = 0;
		lastusedtime = 0;
	}
};
frame_t frame_table[MAX_FRAMES];
Process Processes[1000];
char *FileName = NULL, *FileName2 = NULL;
int num_frames = 0, OPFS = 0, EOFile = 0;
int ofs = 0, Num = 0;
int *Numbers = NULL;
int ProcessesNum = 0;
char buffer[10000] = { 0 };
char ins1[100000] = { 0 };
int ins2[100000] = { 0 };
long unsigned int insNum = 0;
long unsigned int ctx_switches = 0;
long unsigned int process_exits = 0;
unsigned long long cost = 0;
deque <frame_t*> free_list;
FILE *fp = NULL;
int Count = 0;
Process *current_process;
int myrandom(int burst)
{
	return Numbers[(ofs++) % Num] % burst;
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
class Pager
{
public:
	vector <frame_t*> Q;
	virtual frame_t* select_victim_frame() = 0;
}*THE_PAGER;
class FIFO : public Pager
{
public:
	int i;
	FIFO()
	{
		i = 0;
	}
	frame_t* select_victim_frame()
	{
		frame_t* temp = &(frame_table[i%num_frames]);
		i++;
		return temp;
	}
};
class Random : public Pager
{
public:
	vector <frame_t*> Q;
	frame_t* select_victim_frame()
	{
		int i = myrandom(num_frames);
		frame_t* temp = &(frame_table[i]);
		return temp;
	}
};
class Clock : public Pager
{
public:
	int i;
	Clock()
	{
		i = 0;
	}
	frame_t* select_victim_frame()
	{
		frame_t* temp = NULL;
		while (1)
		{
			temp = &(frame_table[i%num_frames]);
			if (Processes[temp->mappedProcess].page_table[temp->mappedvpage].REFERENCED == 1)
			{
				Processes[temp->mappedProcess].page_table[temp->mappedvpage].REFERENCED = 0;
				i++;
			}
			else
			{
				i++;
				return temp;
			}
		}
	}
};
class ESC_NRU : public Pager
{
public:
	int i;
	int inscount;
	int c[4];
	int cl;
	void set()
	{
		for (int k = 0; k < 4; k++) c[k] = -1;
	}
	ESC_NRU()
	{
		set();
		i = 0;
		inscount = Count;
	}
	frame_t* select_victim_frame()
	{
		frame_t* temp = NULL;
		set();
		if (Count - inscount >= 50)
		{
			for (int k = 0; k < num_frames; k++)
			{
				temp = &(frame_table[(i + k) % num_frames]);
				cl = Processes[temp->mappedProcess].page_table[temp->mappedvpage].REFERENCED * 2 + Processes[temp->mappedProcess].page_table[temp->mappedvpage].MODIFIED;
				if (c[cl] == -1)
				{
					c[cl] = (i + k) % num_frames;
				}
				Processes[temp->mappedProcess].page_table[temp->mappedvpage].REFERENCED = 0;
			}
			for (int k = 0; k < 4; k++)
			{
				if (c[k] != -1)
				{
					inscount = Count;
					i = c[k] + 1;
					return &(frame_table[c[k]]);
				}
			}
			return NULL;
		}
		else
		{
			for (int k = 0; k < num_frames; k++)
			{
				temp = &(frame_table[(i + k) % num_frames]);
				cl = Processes[temp->mappedProcess].page_table[temp->mappedvpage].REFERENCED * 2 + Processes[temp->mappedProcess].page_table[temp->mappedvpage].MODIFIED;
				if (c[cl] == -1) c[cl] = (i + k) % num_frames;
			}
			for (int k = 0; k < 4; k++)
			{
				if (c[k] != -1)
				{
					i = c[k] + 1;
					return &(frame_table[c[k]]);
				}
			}
			return NULL;
		}
	}
};
class AGING : public Pager
{
public:
	int i, a;
	unsigned int min;
	int index;
	AGING()
	{
		i = 0;
		min = 0xffffffff;
		a = 0;
	}
	frame_t* select_victim_frame()
	{
		for (int k = 0; k < num_frames; k++)
		{
			frame_table[(i + k) % num_frames].age = frame_table[(i + k) % num_frames].age >> 1;
			if (Processes[frame_table[(i + k) % num_frames].mappedProcess].page_table[frame_table[(i + k) % num_frames].mappedvpage].REFERENCED == 1)
			{
				frame_table[(i + k) % num_frames].age = frame_table[(i + k) % num_frames].age | 0x80000000;
				Processes[frame_table[(i + k) % num_frames].mappedProcess].page_table[frame_table[(i + k) % num_frames].mappedvpage].REFERENCED = 0;
			}
			if (min > frame_table[(i + k) % num_frames].age)
			{
				a = 1;
				min = frame_table[(i + k) % num_frames].age;
				index = (i + k) % num_frames;
			}
		}
		i = (index + 1) % num_frames;
		min = 0xffffffff;
		a = 0;
		return &(frame_table[index]);
	}
};
class WORKING_SET : public Pager
{
public:
	int i;
	WORKING_SET()
	{
		i = 0;
	}
	frame_t* select_victim_frame()
	{
		unsigned int min = 0xffffffff;
		int index = -1;
		frame_t* temp = NULL;
		aaa:
		for (int k = 0; k < num_frames; k++)
		{
			if (Processes[frame_table[(i + k) % num_frames].mappedProcess].page_table[frame_table[(i + k) % num_frames].mappedvpage].REFERENCED == 1)
			{
				frame_table[(i + k) % num_frames].lastusedtime = Count;
				Processes[frame_table[(i + k) % num_frames].mappedProcess].page_table[frame_table[(i + k) % num_frames].mappedvpage].REFERENCED = 0;
			}
			else
			{
				if (Count - frame_table[(i + k) % num_frames].lastusedtime > WS)
				{
					temp = &(frame_table[(i + k) % num_frames]);
					i = (i + k) % num_frames + 1;
					return temp;
				}
				else
				{
					if (min > frame_table[(i + k) % num_frames].lastusedtime)
					{
						min = frame_table[(i + k) % num_frames].lastusedtime;
						index = (i + k) % num_frames;
					}
				}
			}
		}
		if(index==-1) goto aaa;
		i = index + 1;
		return &(frame_table[index]);
	}
};
void InitFreeList()
{
	for (int i = 0; i<num_frames; i++)
	{
		if (frame_table[i].mapped == 0)
		{
			frame_table[i].fid = i;
			free_list.push_back(&(frame_table[i]));
		}
	}
}
void AddFreeList(frame_t* temp)
{
	free_list.push_back(temp);
}
void Createbuffer()
{
	char* i;
	if (fp == NULL) fp = fopen(FileName, "r");
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
void CreateProcessesIns()
{
	int temp = 0;
	Createbuffer();
	sscanf(buffer, "%d", &ProcessesNum);
	for (int i1 = 0; i1<ProcessesNum; i1++)
	{
		Processes[i1].pid = i1;
		Createbuffer();
		sscanf(buffer, "%d", &Processes[i1].VMANum);
		Processes[i1].VMA = (VMA1*)malloc(sizeof(VMA1)*Processes[i1].VMANum);
		for (int i2 = 0; i2<Processes[i1].VMANum; i2++)
		{
			Createbuffer();
			sscanf(buffer, "%d %d %d %d", &Processes[i1].VMA[i2].start_vpage, &Processes[i1].VMA[i2].end_vpage, &Processes[i1].VMA[i2].write_protected, &Processes[i1].VMA[i2].file_mapped);
		}
	}
	while (1)
	{
		Createbuffer();
		if (EOFile == 1) break;
		else
		{
			sscanf(buffer, "%c %d", &ins1[temp], &ins2[temp]);
			temp++;
		}
	}
	insNum = temp;
	fclose(fp);
}
frame_t* allocate_frame_from_free_list()
{
	frame_t* temp = NULL;
	if (free_list.size() == 0) return NULL;
	temp = free_list.front();
	free_list.pop_front();
	return temp;
}
frame_t* get_frame()
{
	frame_t *frame = allocate_frame_from_free_list();
	if (frame == NULL) frame = THE_PAGER->select_victim_frame();
	return frame;
}
int get_next_instruction(char &operation, int &vpage)
{
	if (Count<insNum)
	{
		operation = (ins1[Count]);
		vpage = (ins2[Count]);
		Count++;
		return 1;
	}
	else return 0; 
}
int InVMA(Process* temp, int vpage)
{
	for (int i = 0; i<temp->VMANum; i++)
	{
		if (vpage <= temp->VMA[i].end_vpage && vpage >= temp->VMA[i].start_vpage)
		return i;
	}
	return -1;
}
void Simulation()
{
	char operation;
	int vpage;
	int currentVMA = -1, lastVMA = -1;
	int temp, temp1;
	while (get_next_instruction(operation, vpage))
	{
		printf("%d: ==> %c %d\n", Count - 1, operation, vpage);
		if (operation == 'c')
		{
			current_process = &(Processes[vpage]);
			ctx_switches++;
			cost += COST_CTW;
			continue;
		}
		else if (operation == 'e')
		{
			printf("EXIT current process %d\n", vpage);
			process_exits++;
			cost += COST_E;
			for (int i = 0; i<PTENUM; i++)
			{
				if (current_process->page_table[i].PRESENT_VALID)
				{
					printf(" UNMAP %d:%d\n", current_process->pid, i);
					cost += COST_MAPUNMAP;
					current_process->stats.unmaps++;
					temp = InVMA(current_process, i);
					if (current_process->page_table[i].MODIFIED == 1 && current_process->VMA[temp].file_mapped == 1)
					{
						printf(" FOUT\n");
						cost += COST_FINFOUT;
						current_process->stats.fouts++;
					}
					frame_table[current_process->page_table[i].frames].mappedProcess = -1;
					frame_table[current_process->page_table[i].frames].mappedvpage = -1;
					frame_table[current_process->page_table[i].frames].mapped = 0;
					AddFreeList(&(frame_table[current_process->page_table[i].frames]));
				}
				current_process->page_table[i].frames = 0;
				current_process->page_table[i].MODIFIED = 0;
				current_process->page_table[i].PAGEDOUT = 0;
				current_process->page_table[i].PRESENT_VALID = 0;
				current_process->page_table[i].REFERENCED = 0;
				current_process->page_table[i].set = 0;
				current_process->page_table[i].sw = 0;
				current_process->page_table[i].used = 0;
				current_process->page_table[i].WRITE_PROTECT = 0;
			}
			continue;
		}
		currentVMA = InVMA(current_process, vpage);
		if (currentVMA == -1)
		{
			cost += COST_RW;
			printf(" SEGV\n");
			cost += COST_SEGV;
			current_process->stats.segv++;
			continue;
		}
		pte_t *pte = &(current_process->page_table[vpage]);
		pte->used = 1;
		if (!pte->PRESENT_VALID)
		{

			frame_t *newframe = get_frame();
			if(newframe->mappedProcess!=-1)
			lastVMA = InVMA(&Processes[newframe->mappedProcess], newframe->mappedvpage);
			else lastVMA = -1;
			if (newframe->mapped == 1)
			{
				printf(" UNMAP %d:%d\n", newframe->mappedProcess, newframe->mappedvpage);
				cost += COST_MAPUNMAP;
				Processes[newframe->mappedProcess].stats.unmaps++;
				Processes[newframe->mappedProcess].page_table[newframe->mappedvpage].PRESENT_VALID = 0;
				newframe->mapped = 0;
			}
			if (newframe->mappedProcess != -1 && Processes[newframe->mappedProcess].page_table[newframe->mappedvpage].MODIFIED == 1)
			{
				if (Processes[newframe->mappedProcess].VMA[lastVMA].file_mapped == 1)
				{
					printf(" FOUT\n");
					cost += COST_FINFOUT;
					Processes[newframe->mappedProcess].stats.fouts++;
				}
				else
				{
					printf(" OUT\n");
					cost += COST_INOUT;
					Processes[newframe->mappedProcess].stats.outs++;
					Processes[newframe->mappedProcess].page_table[newframe->mappedvpage].PAGEDOUT = 1;
				}
					
				Processes[newframe->mappedProcess].page_table[newframe->mappedvpage].MODIFIED = 0;
				Processes[newframe->mappedProcess].page_table[newframe->mappedvpage].sw = 1;
			}
			if (current_process->VMA[currentVMA].file_mapped == 1)
			{
				printf(" FIN\n");
				cost += COST_FINFOUT;
				current_process->stats.fins++;
				pte->sw = 0;
			}
			else if (pte->PAGEDOUT == 0)//sw?
			{
				printf(" ZERO\n");
				cost += COST_Z;
				current_process->stats.zero++;
			}
			else if (pte->PAGEDOUT == 1)//sw?
			{
				printf(" IN\n");
				cost += COST_INOUT;
				current_process->stats.ins++;
				pte->sw = 0;
			}
			if (operation == 'r')
			{
				cost += COST_RW;
				pte->REFERENCED = 1;
			}
			else if (operation == 'w')
			{
				cost += COST_RW;
				if (pte->set == 0)
				{
					temp1 = InVMA(current_process, vpage);
					pte->WRITE_PROTECT = current_process->VMA[temp1].write_protected;
					pte->set = 1;
				}
			}
			pte->frames = newframe->fid;
			pte->PRESENT_VALID = 1;
			newframe->mapped = 1; 
			newframe->mappedProcess = current_process->pid;
			newframe->mappedvpage = vpage;
			newframe->lastusedtime=Count;
			newframe->age = 0;
			printf(" MAP %d\n", newframe->fid);
			cost += COST_MAPUNMAP;
			current_process->stats.maps++;
			if (operation == 'w' && pte->WRITE_PROTECT == 1)
			{
				printf(" SEGPROT\n");
				cost += COST_SEGPROT;
				current_process->stats.segprot++;
				pte->REFERENCED = 1;
				pte->MODIFIED = 0;
			}
			else if(operation == 'w' && pte->WRITE_PROTECT == 0)
			{
				pte->REFERENCED = 1;
				pte->MODIFIED = 1;
			}
		}
		else
		{
			frame_table[pte->frames].lastusedtime=Count;
			if (operation == 'r')
			{
				cost += COST_RW;
				pte->REFERENCED = 1;
			}
			else if (operation == 'w')
			{
				cost += COST_RW;
				if (pte->set == 0)
				{
					temp1 = InVMA(current_process, vpage);
					pte->WRITE_PROTECT = current_process->VMA[temp1].write_protected;
					pte->set = 1;
				}
				if (pte->WRITE_PROTECT == 1)
				{
					printf(" SEGPROT\n");
					cost += COST_SEGPROT;
					current_process->stats.segprot++;
					pte->REFERENCED = 1;
					pte->MODIFIED = 0;
				}
				else
				{
					pte->REFERENCED = 1;
					pte->MODIFIED = 1;
				}
			}
		}

	}
	for (int i = 0; i < ProcessesNum; i++)
	{
		printf("PT[%d]: ", i);
		for (int k = 0; k < PTENUM; k++)
		{
			if (Processes[i].page_table[k].PRESENT_VALID == 0)
			{
				if (Processes[i].page_table[k].PAGEDOUT == 0) printf("* ");
				else printf("# ");
			}
			else
			{
				printf("%d:", k);
				if (Processes[i].page_table[k].REFERENCED == 1) printf("R");
				else printf("-");
				if (Processes[i].page_table[k].MODIFIED == 1) printf("M");
				else printf("-");
				if (Processes[i].page_table[k].PAGEDOUT == 1) printf("S ");//sw?
				else printf("- ");
			}
		}
		printf("\n");
	}
	printf("FT: ");
	for (int i = 0; i < num_frames; i++)
	{
		if (frame_table[i].mapped) printf("%d:%d ", frame_table[i].mappedProcess, frame_table[i].mappedvpage);
		else printf("* ");
	}
	printf("\n");
	for (int i = 0; i < ProcessesNum; i++)
	{
		printf("PROC[%d]: U=%lu M=%lu I=%lu O=%lu FI=%lu FO=%lu Z=%lu SV=%lu SP=%lu\n", i, Processes[i].stats.unmaps, Processes[i].stats.maps, Processes[i].stats.ins, Processes[i].stats.outs,	Processes[i].stats.fins, Processes[i].stats.fouts, Processes[i].stats.zero,	Processes[i].stats.segv, Processes[i].stats.segprot);
	}
	printf("TOTALCOST %lu %lu %lu %llu", insNum, ctx_switches, process_exits, cost);
}
int main(int argc, char **argv)
{
	char *s_value = NULL;
	int inst;
	while ((inst = getopt(argc, argv, "f:a:o:")) != -1)
	{
		switch (inst)
		{
		case 'f':
			s_value = optarg;
			num_frames = atoi(s_value);
			break;
		case 'a':
			s_value = optarg;
			switch (s_value[0])
			{
			case 'f':
				THE_PAGER = new FIFO;
				break;
			case 'r':
				THE_PAGER = new Random;
				break;
			case 'c':
				THE_PAGER = new Clock;
				break;
			case 'e':
				THE_PAGER = new ESC_NRU;
				break;
			case 'a':
				THE_PAGER = new AGING;
				break;
			case 'w':
				THE_PAGER = new WORKING_SET;
				break;
			}
			break;
		case 'o':
			s_value = optarg;
			if (!strcmp(s_value, "OPFS"))
			{
				OPFS = 1;
			}
			break;
		case '?':
			break;
		default:
			break;
		}
	}
	FileName = argv[optind];
	FileName2 = argv[optind + 1];
	CreateProcessesIns();
	CreaterandomNumbers();
	InitFreeList();
	Simulation();
}
