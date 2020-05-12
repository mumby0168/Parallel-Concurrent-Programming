#pragma once

#ifndef TYPESH
#define TYPESH


enum ColorMode
{
	Speed,
	CenterMass,
	Solid
};

struct SimulationParams
{
	ColorMode colorMode;
	float max;
	float dt;

};

#endif