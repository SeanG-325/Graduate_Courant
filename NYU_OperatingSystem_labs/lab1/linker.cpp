#ifndef _CRT_SECURE_NO_WARNINGS
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <ctype.h>

#define MAX_DEFCOUNT 16 
#define MAX_USECOUNT 16 
#define MAX_VARLEN 16
#define MAX_DEFLEN 16
#define MAX_USELEN 16
#define MAX_INSTCOUNT 512
#define MACHINE_SIZE 512

using namespace std;
const char* delim = " \t\n";
char *temp = NULL, *p = NULL, *p1 = NULL, *p2 = "";
int epos = 0, eline = 0;
int module[20] = { 0 };//module offsets
bool eof = false;
int Sym1[300][3] = { 0 };//Sym1[][0]:value, Sym1[][1]:multi-define, Sym1[][2]:module
char Sym2[300][100] = { 0 };//Symtable
int Sym3[300] = { 0 };//sym used
int SymCount = 0;
int Line = 0, Num = 0, Offset = 0;
char *buffer = NULL, *buffer1 = NULL;
FILE *fp = NULL;
int linenum, lineoffset;
int ModuleSize = 0;
char Name[100] = { 0 };
void __parseerror(int errcode)
{
	static char* errstr[] =
	{
		"NUM_EXPECTED", // Number expect
		"SYM_EXPECTED", // Symbol Expected
		"ADDR_EXPECTED", // Addressing Expected which is A/E/I/R
		"SYM_TOO_LONG", // Symbol Name is too long
		"TOO_MANY_DEF_IN_MODULE", // > 16
		"TOO_MANY_USE_IN_MODULE", // > 16
		"TOO_MANY_INSTR", // total num_instr exceeds memory size (512)
	};
	printf("Parse Error line %d offset %d: %s\n", linenum, lineoffset, errstr[errcode]);
	exit(errcode);
}
void Error(int errcode, int value = 0, char *s = "")
{
	switch (errcode)
	{
	case 0:
		if (value == -1)
		{
			__parseerror(0);
		}
		break;
	case 1:
		if (s[0] == 0)
		{
			__parseerror(1);
		}
		break;
	case 2:
		if (s[0] == 0)
		{
			__parseerror(2);
		}
		break;
	case 3:
		if (strlen(s)>MAX_VARLEN)
		{
			__parseerror(3);
		}
		break;
	case 4:
		if (value>MAX_DEFLEN)
		{
			__parseerror(4);
		}
		break;
	case 5:
		if (value>MAX_USELEN)
		{
			__parseerror(5);
		}
		break;
	case 6:
		if (value>MAX_INSTCOUNT)
		{
			__parseerror(6);
		}
		break;
	default:
		break;
	}
}
char* getToken()
{
	if (fp == NULL)
	{
		fp = fopen(Name, "r");
	}
	if (p2[0] == 0 || p2[0] == '\n')
	{
	newline:
		if (!fgets(buffer, 100, fp))//eof
		{
			linenum = eline;
			lineoffset = epos;
			eof = true;
			return "";
		}
		else
		{
			linenum++;
			strcpy(buffer1, buffer);
			if (strlen(buffer) != 0)
			{
				eline = linenum;
				epos = strlen(buffer);
			}
			else if (epos == 0)
			{
				eline = linenum;
				epos = 1;
			}
			temp = strtok(buffer, delim);
			p2 = buffer1;
			if (temp)
			{
				p = strstr(p2, temp);
				p1 = strstr(buffer1, p);
				p2 = p;
				p2 += strlen(temp);
				lineoffset = p1 - buffer1 + 1;
				return temp;
			}
			else
			{
				goto newline;
			}
		}
	}
	else
	{
		temp = strtok(NULL, delim);
		if (temp)
		{
			p = strstr(p2, temp);
			p1 = strstr(buffer1, p);
			p2 = p;
			p2 += strlen(temp);
			lineoffset = p1 - buffer1 + 1;
			return temp;
		}
		else
		{
			goto newline;
		}
	}
}
int readInt()
{
	char *ans = getToken();
	int i;
	if (strlen(ans) == 0)//eof
	{
		return -1;
	}
	else if (strspn(ans, "0123456789") != strlen(ans))
	{
		return -1;
	}
	else
	{
		i = atoi(ans);
		return i;
	}
}
char* readSym()
{
	char *ans = getToken();
	if (!isalpha(ans[0]))
	{
		return "";
	}
	else
	{
		for (int i = 1; i<strlen(ans); i++)
		{
			if (!isalnum(ans[i]))
			{
				return "";
			}
		}
		return ans;
	}
}
char readIEAR()
{
	char *ans = getToken();
	if (strlen(ans) == 1)
	{
		if (ans[0] == 'I')
		{
			return 'I';
		}
		else if (ans[0] == 'E')
		{
			return 'E';
		}
		else if (ans[0] == 'A')
		{
			return 'A';
		}
		else if (ans[0] == 'R')
		{
			return 'R';
		}
		else
		{
			return 0;
		}
	}
	else
	{
		return 0;
	}
}
void NewSym(char sym[], int val, int m)
{
	for (int i = 0; i<SymCount; i++)
	{
		if (!strcmp(sym, Sym2[i]))
		{
			Sym1[i][1] = 1;//sign multi-def on Sym1[][1], Sym1[][0] is the first value
			return;
		}
	}
	strcpy(Sym2[SymCount], sym);
	Sym1[SymCount][0] = val;
	Sym1[SymCount][2] = m;
	SymCount++;
}
void Pass1()
{
	int i = 0, j = 0, m = 0, a = 0, sum_inst = 0;
	int instcount = 0;
	int val = 0;
	char sym[100] = { 0 };
	while (1)
	{
		module[m++] = sum_inst;
		int defcount = readInt();
		if (eof == true)
		{
			break;
		}
		Error(0, defcount);
		Error(4, defcount);
		for (int k = 0; k<defcount; k++)
		{
			strcpy(sym, readSym());//
			Error(1, 0, sym);
			Error(3, 0, sym);
			val = readInt();//
			Error(0, val);
			NewSym(sym, val + module[m - 1], m);
		}
		int usecount = readInt();//
		Error(0, usecount);
		Error(5, usecount);
		for (int k = 0; k<usecount; k++)
		{
			strcpy(sym, readSym());//
			Error(1, 0, sym);
			Error(3, 0, sym);
		}
		instcount = readInt();//Sum?
		sum_inst += instcount;
		Error(0, instcount);
		Error(6, sum_inst);
		for (int k = 0; k<instcount; k++)
		{
			char addressmode = readIEAR();
			Error(2, 0, &addressmode);
			int operand = readInt();
			Error(0, operand);
		}
		for (int k = 0; k<SymCount; k++)
		{
			if (Sym1[k][0] >= sum_inst)
			{
				printf("Warning: Module %d: %s too big %d (max=%d) assume zero relative\n", m, Sym2[k], Sym1[k][0] - module[m - 1], instcount - 1);
				Sym1[k][0] = module[m - 1];
			}
		}
	}
	ModuleSize = m;
}
void Pass2()
{
	int i = 0, j = 0, m = 0, a = 0, sum_inst = 0;
	int val = 0;
	char uselist[20][20] = { 0 };
	int used[20] = { 0 };
	char sym[100] = { 0 };
	int ansline = 0;
	printf("Symbol Table\n");
	for (int k = 0; k<SymCount; k++)
	{
		printf("%s=%d", Sym2[k], Sym1[k][0]);
		if (Sym1[k][1] == 1)
		{
			printf(" Error: This variable is multiple times defined; first value used");
		}
		printf("\n");
	}
	printf("\nMemory Map\n");
	while (1)
	{
		for (int k = 0; k<20; k++)
		{
			uselist[k][0] = 0;
			used[k] = 0;
		}
		a = 0;
		module[m++] = sum_inst;
		int defcount = readInt();
		if (eof == true)
		{
			break;
		}
		for (int k = 0; k<defcount; k++)
		{
			strcpy(sym, readSym());//
			val = readInt();//
		}
		int usecount = readInt();//
		for (int k = 0; k<usecount; k++)
		{
			strcpy(sym, readSym());//
								   //
			strcpy(uselist[a++], sym);
		}
		int instcount = readInt();//Sum?
		sum_inst += instcount;
		for (int k = 0; k<instcount; k++)
		{
			printf("%03d: ", ansline++);
			char addressmode = readIEAR();
			//
			int operand = readInt();
			int opcode = operand / 1000;
			int addr = operand % 1000;
			int find = -1;
			switch (addressmode)
			{
			case 'I':
				if (operand >= 10000)
				{
					printf("9999 Error: Illegal immediate value; treated as 9999\n");
				}
				else
				{
					printf("%04d\n", operand);
				}
				break;
			case 'E':
				if (opcode >= 10)
				{
					printf("9999 Error: Illegal opcode; treated as 9999\n");
				}
				else if (addr >= usecount)
				{
					printf("%04d Error: External address exceeds length of uselist; treated as immediate\n", operand);
				}
				else
				{
					used[addr] = 1;
					for (int t = 0; t<SymCount; t++)
					{
						if (!strcmp(uselist[addr], Sym2[t]))
						{
							find = t;
							Sym3[find] = 1;
							break;
						}
					}
					if (find != -1)
					{
						int ans = opcode * 1000 + Sym1[find][0];
						printf("%04d\n", ans);
					}
					else
					{
						int ans = opcode * 1000;
						printf("%04d Error: %s is not defined; zero used\n", ans, uselist[addr]);
					}
				}
				break;
			case 'A':
				if (opcode >= 10)
				{
					printf("9999 Error: Illegal opcode; treated as 9999\n");
				}
				else if (addr >= MACHINE_SIZE)
				{
					int ans = opcode * 1000;
					printf("%04d Error: Absolute address exceeds machine size; zero used\n", ans);
				}
				else
				{
					printf("%04d\n", operand);
				}
				break;
			case 'R':
				if (opcode >= 10)
				{
					printf("9999 Error: Illegal opcode; treated as 9999\n");
				}
				else if (addr >= instcount)
				{
					int ans = opcode * 1000 + module[m - 1];
					printf("%04d Error: Relative address exceeds module size; zero used\n", ans);
				}
				else
				{
					int ans = opcode * 1000 + module[m - 1] + addr;
					printf("%04d\n", ans);
				}
				break;
			default:
				break;
			}
		}
		for (int k = 0; k<usecount; k++)
		{
			if (used[k] == 0)
			{
				printf("Warning: Module %d: %s appeared in the uselist but was not actually used\n", m, uselist[k]);
			}
			else
			{
				for (int t = 0; t<SymCount; t++)
			    {
				    if (!strcmp(uselist[k], Sym2[t]))
				    {
					    Sym3[t] = 1;
					    break;
			    	}
		    	}
			}
		}
	}
	printf("\n");
	for (int k = 0; k<SymCount; k++)
	{
		if (Sym3[k] == 0)
		{
			printf("Warning: Module %d: %s was defined but never used\n", Sym1[k][2], Sym2[k]);
		}
	}
}
int main(int argc, char** argv)
{
	buffer = (char*)malloc(sizeof(char) * 200);
	buffer1 = (char*)malloc(sizeof(char) * 200);
	strcpy(Name, argv[1]);

	Pass1();

	temp = NULL;
	p = NULL;
	p1 = NULL;
	p2 = "";
	linenum = 0;
	lineoffset = 0;
	fclose(fp);
	fp = NULL;
	eof = false;

	Pass2();

	fclose(fp);
	free(buffer);
	free(buffer1);
}
