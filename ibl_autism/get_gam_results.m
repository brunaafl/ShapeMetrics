function [k, list_area, list_cohort, list_sig, list_pval, model_rate_Hz, x_rate_Hz, raw_rate_Hz, full_pseudo_r2_eval, mutual_info, d_prime, kernel_Hz, kernel_x, kernel_mCI, kernel_pCI, kernel_Hz_mCI, kernel_Hz_pCI, name_info] = get_gam_fit_results_and_name(data_dir, variable, coupling_flag)

r_cutoff = 0.01; % how good the fit to spikes are
counter = 0;
sig_cutoff = 0.001; % what we will call significant
name_info = struct();


for i_data = 1:size(data_dir, 1)
    
    cohort_dir = dir(fullfile(data_dir(i_data).folder, data_dir(i_data).name));
    cohort_dir = cohort_dir(~ismember({cohort_dir.name},{'.','..', '.DS_Store'}));
    
    cohort_dir = cohort_dir(ismember({cohort_dir.name}, keep_areas));
    
    for i_area = 1:size(cohort_dir)
        
        area_dir = dir(fullfile(cohort_dir(i_area).folder, cohort_dir(i_area).name));
        area_dir = area_dir(~ismember({area_dir.name},{'.','..', '.DS_Store'}));
        
        base = area_dir.folder;
        
        if coupling_flag
            c0 = dir(fullfile(base, 'gam_fit_useCoupling1_useSubPrior1_*.mat'));
        else
            c0 = dir(fullfile(base, 'gam_fit_useCoupling0_useSubPrior1_*.mat'));
        end
        
        if ~isempty(c0)
            
            for i_file = 1:size(c0, 1)
                
                load(fullfile(c0(i_file).folder, c0(i_file).name));
                fullfile(c0(i_file).folder, c0(i_file).name)
                
                if results(1).full_pseudo_r2_eval > r_cutoff % other option here is train, so eval is the right thing to look at
                    idx = find(strcmp({results.variable}, variable));
                    
                    if ~isempty(idx)
                        
                        %results(idx).mutual_info
                        if ~isnan(results(idx).mutual_info)
                            if size(idx, 2) == 1
                                
                                counter = counter + 1;
                                
                                d_prime(counter) = abs(temp_model - temp_raw)/(temp_model + temp_raw);
                                
                                % kernel
                                k(counter, :) = results(idx).kernel;
                                kernel_mCI(counter, :) = results(idx).kernel_mCI;
                                kernel_pCI(counter, :) = results(idx).kernel_pCI;
                                
                                % make these cell arrays
                                list_area{counter} = cohort_dir(i_area).name;
                                list_cohort{counter} = data_dir(i_data).name;
                                
                                % keep track of significant or not..
                                list_sig(counter) = results(idx).pval < sig_cutoff;
                                list_pval(counter) = results(idx).pval;
                                
                                % other stuff that we can use to filter by
                                model_rate_Hz(counter, :) = results(idx).model_rate_Hz;
                                x_rate_Hz(counter, :) = results(idx).x_rate_Hz;
                                raw_rate_Hz(counter, :) = results(idx).raw_rate_Hz;
                                
                                
                                sigma = (results(idx).kernel_pCI - results(idx).kernel)/2.575;
                                kernel_Hz(counter, :) = exp(results(idx).kernel + (0.5.*sigma.^2)).*results(idx).fr;
                                kernel_Hz_mCI(counter, :) = logninv(1 - 0.975, results(idx).kernel + log(results(idx).fr), sigma);
                                kernel_Hz_pCI(counter, :) = logninv(0.975, results(idx).kernel + log(results(idx).fr), sigma);
                                kernel_x(counter, :) = results(idx).kernel_x;
                                
                                full_pseudo_r2_eval(counter) = results(idx).full_pseudo_r2_eval;
                                mutual_info(counter) = results(idx).mutual_info;
                                
                                name_info(counter).folder = fullfile('D:\MOUSE-ASD-NEURONS\data\4step_v4\data', data_dir(i_data).name, results(idx).brain_area_group);
                                name_info(counter).file = [results(idx).brain_area_group, '_', results(idx).animal_name, '_', results(idx).date, '_', results(idx).session_num];
                                name_info(counter).neuron_id = results(idx).neuron_id;
                            end
                        end
                    end % index is not empty
                end % if the fit is decent...
            end % all the files
            clear c0
        end % if c0 is not empty...
    end % end of area
end % end of cohort
end % end of function..
