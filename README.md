# NANDnet

NANDnet is a new gatenet model that was trained to learn the NAND gate. It can accurately predict the right binary output of give input tokens.

NAND works like this; It returns 0 (FALSE)<br> when both inputs are 1.

Example of NANDnet replying using ```inference.py```

```txt
>> 0
>> 0
nandnet: 1
>> 1
>> 0
nandnet: 1
>> 0
>> 1
nandnet: 1
>> 1
>> 1
nandnet: 0
>> 
```

<footer>
&copy; gatenets | im-lemon | Lode Brusten
</footer>