%{
  ---------------------------------------------------------------------------
    Name       : "depression.m"
    
    Description: calculates the amount of RRAM conductance decrease for a given
                 RRAM characteristics, incase of weight depression
  
    Authors: Kalkidan Deme Muleta, Bai-Sun Kong
  
    Paper: "RRAM-Based Spiking Neural Network with Target-Modulated
            Spike-Timing-Dependent Plasticity" published on IEEE Transactions on
            Biomedical Circuits and Systems.
%}

function [deltaGp] = depression(alpha_m, beta_m, Gold, Gmax, Gmin)
    
    deltaGp = alpha_m*exp(-beta_m*(Gmax-Gold)./(Gmax-Gmin));
end