// Orchard (Obstgarten) kid's game
// Corresponds exactly to the Stormvogel model
pomdp

observables
   s, d, raven
endobservables

observable "hasApple" = apple>0;
observable "hasCherry" = cherry>0;
observable "hasPlum" = plum>0;
observable "hasPear" = pear>0;


// constants
const int NUM_FRUIT=4; //=10; // Number of fruits per fruit type
const int DISTANCE_RAVEN=5; //=9; // Distance of raven from orchard
const double probRaven = 1/6;
const double probBucket = 1/6;
const double probFruit = 1 - probRaven - probBucket;

// Module for the player
module player0
	// Game states of the player
	// s=0 : roll dice
	// s=1 : perform action (pick fruit, choose fruit for 'bucket', move raven)
	s : [0..2] init 2;
	// Dice outcome
	// 0: not thrown, 1: apple, 2:pear, 3:cherry, 4:plum, 5:bucket, 6:raven
	d : [0..6] init 1;

	// Perform stealing
	[steal] (s=2) -> (s'=0);
	// Perform actions
	// Throw dice
	[nextRound] (s=0) & (!game_ended) -> 1/4 * probFruit : (s'=1) & (d'=1)
	                                   + 1/4 * probFruit : (s'=1) & (d'=2)
	                                   + 1/4 * probFruit : (s'=1) & (d'=3)
	                                   + 1/4 * probFruit : (s'=1) & (d'=4)
	                                   + probBucket      : (s'=1) & (d'=5)
	                                   + probRaven       : (s'=1) & (d'=6)
	                                   ;
	[nextRound] (s=0) & ( game_ended) -> 1: true;

	// Pick fruit
	[pickAPPLE]  (s=1) & (d=1) -> 1: (s'=0) & (d'=0);
	[pickPEAR]   (s=1) & (d=2) -> 1: (s'=0) & (d'=0);
	[pickCHERRY] (s=1) & (d=3) -> 1: (s'=0) & (d'=0);
	[pickPLUM]   (s=1) & (d=4) -> 1: (s'=0) & (d'=0);

	// Choose fruit (because of 'bucket')
	[chooseAPPLE]  (s=1) & (d=5) -> 1 : (s'=0) & (d'=0);
	[choosePEAR]   (s=1) & (d=5) -> 1 : (s'=0) & (d'=0);
	[chooseCHERRY] (s=1) & (d=5) -> 1 : (s'=0) & (d'=0);
	[choosePLUM]   (s=1) & (d=5) -> 1 : (s'=0) & (d'=0);
	// No fruit can be chosen
	[nochoice]     (s=1) & (d=5) & (all_trees_empty)
		                         -> 1 : (s'=0) & (d'=0);

	// Move raven
	[moveRaven] (s=1) & (d=6) -> 1 : (s'=0) & (d'=0);
endmodule

// Module for the orchard with the trees
module orchard
	// Fruits
	apple : [0..NUM_FRUIT] init NUM_FRUIT;
	pear : [0..NUM_FRUIT] init NUM_FRUIT;
	cherry : [0..NUM_FRUIT] init NUM_FRUIT;
	plum : [0..NUM_FRUIT] init NUM_FRUIT;

	// Pick fruit
	[pickAPPLE]  (apple>0)  -> 1: (apple'=apple-1);
	[pickAPPLE]  (apple=0)  -> 1: true;
	[pickPEAR]   (pear>0)   -> 1: (pear'=pear-1);
	[pickPEAR]   (pear=0)   -> 1: true;
	[pickCHERRY] (cherry>0) -> 1: (cherry'=cherry-1);
	[pickCHERRY] (cherry=0) -> 1: true;
	[pickPLUM]   (plum>0)   -> 1: (plum'=plum-1);
	[pickPLUM]   (plum=0)   -> 1: true;
	[steal] true -> 1/4: (apple'=apple-1) + 1/4: (pear'=pear-1) + 1/4: (cherry'=cherry-1) 1/4: (plum'=plum-1);

	// Choose fruit
	[chooseAPPLE]  (apple>0)  -> 1: (apple'=apple-1);
	[choosePEAR]   (pear>0)   -> 1: (pear'=pear-1);
	[chooseCHERRY] (cherry>0) -> 1: (cherry'=cherry-1);
	[choosePLUM]   (plum>0)   -> 1: (plum'=plum-1);
	[noChoice]     (all_trees_empty) -> 1: true;
endmodule

// Module for the raven
module raven
	// Raven
	raven : [0..DISTANCE_RAVEN] init DISTANCE_RAVEN;

	// Move raven
	[moveRaven] (raven>0) -> 1 : (raven'=raven-1);
	[moveRaven] (raven=0) -> 1 : true;
endmodule

// labels
formula all_trees_empty = apple = 0
                        & pear = 0
                        & cherry = 0
                        & plum = 0
                        ;

formula game_ended = all_trees_empty | raven = 0;
label "PlayersWon" =  all_trees_empty & raven > 0;
label "RavenWon"   = !all_trees_empty & raven = 0;


// rewards
rewards "rounds"
	s=0 : 1;
endrewards
