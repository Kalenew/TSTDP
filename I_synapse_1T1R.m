%{
  ---------------------------------------------------------------------------
    Name       : "I_Synapse.m"
    
    Description: calculates the postsynpacit current for a grid of 1T1R
                 synapse
  
    Authors: Kalkidan Deme Muleta, Bai-Sun Kong
  
    Paper: "RRAM-Based Spiking Neural Network with Target-Modulated
            Spike-Timing-Dependent Plasticity" published on IEEE Transactions on
            Biomedical Circuits and Systems.
%}

function I_synapse = I_synapse_1T1R(Cx)
Rreset = 10000;
Rset   = 500;
RselOn = 2400;

Res = RselOn + Rreset - Cx*(Rreset - Rset);

I_synapse = 0.3*(1./Res);
end