.. _submitargs:

Arguments of the input script
==============================


Host and nodes
--------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: wf_args

Task definition
--------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: task_args

Input definition
----------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: inputs_args

Exploration
-----------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: explore_args

Explore task group definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ASE task group
****************
.. dargs:: 
    :module: pfd.exploration.task
    :func: AseTaskGroup.args

CALYPSO task group
****************
.. dargs:: 
    :module: pfd.exploration.task
    :func: CalyTaskGroup.args    
    
Frame selection
---------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: select_confs_args

Labeling
-------------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: label_args

Model training
--------------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: training_args

Evaluate
--------

.. dargs:: 
    :module: pfd.entrypoint.args
    :func: evaluate_args



