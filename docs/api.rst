API
===

Simulation
----------

.. autoclass:: flexsweep.Simulator()
    :members:
    :undoc-members:
    :show-inheritance:

Feature vectors
---------------

.. autofunction:: flexsweep.summary_statistics()
.. autofunction:: flexsweep.fv._process_sims()
.. autofunction:: flexsweep.fv._process_vcf()
.. autofunction:: flexsweep.fv.calculate_stats_simplify()
.. autofunction:: flexsweep.fv.calculate_stats_vcf()

Population genetics statisitcs
------------------------------

Site Frequency Spectrum (SFS)–based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.theta_watterson()
.. autofunction:: flexsweep.fv.sfs_nb()
.. autofunction:: flexsweep.fv.theta_pi()
.. autofunction:: flexsweep.fv.tajima_d()
.. autofunction:: flexsweep.fv.achaz_y()
.. autofunction:: flexsweep.fv.fay_wu_h_norm()
.. autofunction:: flexsweep.fv.zeng_e()
.. autofunction:: flexsweep.fv.fuli_f_star()
.. autofunction:: flexsweep.fv.fuli_f()
.. autofunction:: flexsweep.fv.fuli_d_star()
.. autofunction:: flexsweep.fv.fuli_d()

Derived-background diversity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.dind_high_low()
.. autofunction:: flexsweep.fv.s_ratio()


Haplotype-based
~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.ihs_ihh()
.. autofunction:: flexsweep.fv.nsl()
.. autofunction:: flexsweep.fv.haf_top()
.. autofunction:: flexsweep.fv.run_isafe()
.. autofunction:: flexsweep.fv.garud_h_numba()
.. autofunction:: flexsweep.fv.h12_enard()
.. autofunction:: flexsweep.fv.hapdaf_o()
.. autofunction:: flexsweep.fv.hapdaf_s()


LD
~~

.. autofunction:: flexsweep.fv.Ld()
.. autofunction:: flexsweep.fv.r2()
.. autofunction:: flexsweep.fv.compute_r2_matrix_upper()
.. autofunction:: flexsweep.fv.omega_linear_correct()


Composite methods
~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.compute_t_m()
.. autofunction:: flexsweep.fv.mu_stat()

Balancing selection
~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.balancing.ncd1()
.. autofunction:: flexsweep.balancing.run_beta_window()


Convolutional Neural Network
----------------------------

.. autoclass:: flexsweep.CNN()
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: flexsweep.cnn.DAParquetSequence()
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: flexsweep.cnn.rank_probabilities()







