function knn2
   [data1tr,text1tr]= xlsread('forest_type_train_1',1,'B2:AC55'); %  training set 10% of data
   X1= data1tr(1:54, 1:27);
   y1= text1tr(1:54, 1);      
   for i=1:54
       Y1(i)= strtrim(y1(i));
   end
   Y1=Y1';             % just to remove the space after each class letter
   
   obj1 = fitcknn(X1,Y1,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl1 = crossval(obj1);
   cvmdlloss1 = kfoldLoss(cvmdl1);  %10 fold validation
   crossvalerrorrate1= cvmdlloss1
   
   [data1t,text1t]= xlsread('forest_type_test_1',1,'B2:AC470');
   test1= data1t(1:469, 1:27);
   label1 = predict(obj1,test1);
   actual1original = text1t(1:469, 1);
   for i=1:469
       actual1(i)= strtrim(actual1original(i));
   end
   actual1=actual1';
   error=0;
   
   for i=1:469
     if (strcmp(label1(i),actual1(i))~=1)
         error=error+1;
     end
   end
      errorrate1= error/469
      
    %  training set 20% of data  
   [data2tr,text2tr]= xlsread('forest_type_train_2',1,'B2:AC107'); 
   X2= data2tr(1:106, 1:27);
   y2= text2tr(1:106, 1);      
   for i=1:106
       Y2(i)= strtrim(y2(i));
   end
   Y2=Y2';             % just to remove the space after each class letter
   
   obj2 = fitcknn(X2,Y2,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl2 = crossval(obj2);
   cvmdlloss2 = kfoldLoss(cvmdl2);  %10 fold validation
   crossvalerrorrate2= cvmdlloss2
   
   [data2t,text2t]= xlsread('forest_type_test_2',1,'B2:AC418');
   test2= data2t(1:417, 1:27);
   label2 = predict(obj2,test2);
   actual2original = text2t(1:417, 1);
   for i=1:417
       actual2(i)= strtrim(actual2original(i));
   end
   actual2=actual2';
   error=0;
   
   for i=1:417
     if (strcmp(label2(i),actual2(i))~=1)
         error=error+1;
     end
   end
      errorrate2= error/417  
   
   %  training set 30% of data  
   [data3tr,text3tr]= xlsread('forest_type_train_3',1,'B2:AC159'); 
   X3= data3tr(1:158, 1:27);
   y3= text3tr(1:158, 1);      
   for i=1:158
       Y3(i)= strtrim(y3(i));
   end
   Y3=Y3';             % just to remove the space after each class letter
   
   obj3 = fitcknn(X3,Y3,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl3 = crossval(obj3);
   cvmdlloss3 = kfoldLoss(cvmdl3);  %10 fold validation
   crossvalerrorrate3= cvmdlloss3
   
   [data3t,text3t]= xlsread('forest_type_test_3',1,'B2:AC366');
   test3= data3t(1:365, 1:27);
   label3 = predict(obj3,test3);
   actual3original = text3t(1:365, 1);
   for i=1:365
       actual3(i)= strtrim(actual3original(i));
   end
   actual3=actual3';
   error=0;
   
   for i=1:365
     if (strcmp(label3(i),actual3(i))~=1)
         error=error+1;
     end
   end
      errorrate3= error/365 
      
  %  training set 40% of data  
   [data4tr,text4tr]= xlsread('forest_type_train_4',1,'B2:AC212'); 
   X4= data4tr(1:211, 1:27);
   y4= text4tr(1:211, 1);      
   for i=1:211
       Y4(i)= strtrim(y4(i));
   end
   Y4=Y4';             % just to remove the space after each class letter
   
   obj4 = fitcknn(X4,Y4,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl4 = crossval(obj4);
   cvmdlloss4 = kfoldLoss(cvmdl4);  %10 fold validation
   crossvalerrorrate4= cvmdlloss4
   
   [data4t,text4t]= xlsread('forest_type_test_4',1,'B2:AC313');
   test4= data4t(1:312, 1:27);
   label4 = predict(obj4,test4);
   actual4original = text4t(1:312, 1);
   for i=1:312
       actual4(i)= strtrim(actual4original(i));
   end
   actual4=actual4';
   error=0;
   
   for i=1:312
     if (strcmp(label4(i),actual4(i))~=1)
         error=error+1;
     end
   end
      errorrate4= error/312
          
  %  training set 50% of data  
   [data5tr,text5tr]= xlsread('forest_type_train_5',1,'B2:AC264'); 
   X5= data5tr(1:263, 1:27);
   y5= text5tr(1:263, 1);      
   for i=1:263
       Y5(i)= strtrim(y5(i));
   end
   Y5=Y5';             % just to remove the space after each class letter
   
   obj5 = fitcknn(X5,Y5,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl5 = crossval(obj5);
   cvmdlloss5 = kfoldLoss(cvmdl5);  %10 fold validation
   crossvalerrorrate5= cvmdlloss5
   
   [data5t,text5t]= xlsread('forest_type_test_5',1,'B2:AC261');
   test5= data5t(1:260, 1:27);
   label5 = predict(obj5,test5);
   actual5original = text5t(1:260, 1);
   for i=1:260
       actual5(i)= strtrim(actual5original(i));
   end
   actual5=actual5';
   error=0;
   
   for i=1:260
     if (strcmp(label5(i),actual5(i))~=1)
         error=error+1;
     end
   end
      errorrate5= error/260
              