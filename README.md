###Abstract
In the US, motor vehicle accident deaths are a leading cause of death of children, and a significant cause of death of adults,
including over 40,000 deaths and 2.44 million injuries in 2023. Pedestrians/pedalcyclists make up 22% of deaths,
increasing to 30% in urban areas. By combining multiple datasets, including a crash accident 
dataset containing hundreds of thousands of individual crashes, and Mapillary’s dataset on the location of stop signs 
and traffic signals, we are able to train a model using machine learning to help predict when pedestrian accidents will happen, 
as well as investigate how policy changes can reduce the number of pedestrian accidents.
There are several existing challenges when using accident datasets to make predictions.
First, data integration and alignment present a significant obstacle. The Chicago crash dataset 
and Mapillary’s infrastructure dataset differ in format, scale, temporal resolution, and geographic precision. Matching crash coordinates with nearby traffic signs or signals requires careful spatial preprocessing and geocoding accuracy. Second, data completeness and bias may affect model performance. Not all road segments are equally covered by Mapillary imagery, and crash reports may contain missing or inconsistently recorded variables. This can introduce spatial or demographic bias into the predictive model. Third, causal inference vs. correlation poses an analytical challenge. While machine learning models can detect statistical relationships between signage and pedestrian accidents, distinguishing whether poor signage causes accidents—or merely correlates with high-risk environments—requires careful interpretation and possibly additional controls.
This problem is significant because prior work underscores the importance of road infrastructure. 
While it is quite difficult to change the psychology of the way people drive, i.e. making drivers “safer”, 
it is feasible and potentially cost-effective to update road signs and traffic signals. Additionally, policymakers often consider 
only historical crash counts, rather than predictive analysis to help make investment decisions.
This project proposes the development of an intelligent framework to predict the likelihood of pedestrian-involved 
traffic accidents based on the presence and proximity of road infrastructure. While traditional accident analysis often 
focuses on driver behavior or weather conditions, this study investigates the correlation between physical road markings, traffic signs, and safety outcomes. Using open-source geospatial imagery from Mapillary alongside the Chicago Public Safety dataset, the research employs a dual-model approach. Logistic Regression and Deep Neural Networks (DNN) are utilized to calculate crash probabilities at specific coordinates, while spatial analysis determines if historical accident hotspots suffer from a lack of adequate signage or poor sign placement. Furthermore, the framework serves as a policy simulation tool. By modeling "what-if" scenarios such as the installation of stop signs at high-risk intersections or signalizing the top 5% of accident-prone areas; the project provides quantifiable predictions on how infrastructure interventions can reduce pedestrian fatalities. The final output aims to provide urban planners with a data-driven roadmap for proactive traffic safety management.
