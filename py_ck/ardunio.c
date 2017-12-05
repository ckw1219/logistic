int light = 10;
int ex=11;
int num;
int cod = 0;//腥红/靓青/春天绿/金/午夜蓝/橙红/柠檬绿/火砖/深洋紫/橄榄褐
int a[240] = {0,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,
              0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,
              1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,
              1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
              0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,0,0,0,
              0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
              1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0,
              0,0,1,0,0,0,1,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,
              0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,
              1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,0,0,0,1,1};
char msg = ' ';
void setup()
{
  pinMode(light, OUTPUT);
  pinMode(ex, OUTPUT);
 
  Serial.begin(9600);
}

void loop ()
{
 while(Serial.available()>0){
       msg = Serial.read();
    }
  if(msg == 'Y')
  {
    Serial.print("led Activated\n");
   black ();
     delay (500);
    num = 0;
    while (num != 240)
    {
      cod = a[num];
      switch (cod)
       {
          case 0 : {PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001100;};break;
          case 1 : {PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001100;};break;
        }
        num++;
    }
   PORTB = B001000;
   delay (500);
  }
}
void black ()
{
  int i;
  for (i = 0;i<240;i++)
  {PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001100;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001000;PORTB = B001100;}
  PORTB = B001000;
}
