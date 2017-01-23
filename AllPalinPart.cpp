#include<iostream>
#include<string>
#include<stdlib.h>
//#include<boolean>


using namespace std;


int isPalin(string str,int i,int j)
{
if(i==j)
return true;

while(i<j)
{
    if((str.at(i)!= str.at(j)))
        return false;
    else
        {
            i++;
            j--;

        }

}

return true;
}


void allPalPartitions(string str)
{

int n = str.length();
cout<<str;
cout<<n;
for(int i=0;i<n;i++)
{
    for(int j=i;j<n;j++)
    {
        if(isPalin(str,i,j))
        {
        cout<<str.substr(i,j-i+1)<<"\n";
        }

    }


}

return;
}


// Driver program
int main()
{
	string str = "nitin";
	cout<<str;
	allPalPartitions(str);
	return 0;
}
