.. currentmodule:: pylelemmatize

pylelemmatize
=============

Public API (summary)
--------------------

..
   .. autosummary::
      :toctree: _autosummary
      :nosignatures:



Classes
-------

.. autoclass:: AbstractLemmatizer
   :noindex:
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: LemmatizerBMP
   :noindex:
   :members:
   :inherited-members:
   :show-inheritance:

..
   :exclude-members: from_config  ; remove this line or edit as you like

.. autoclass:: GenericLemmatizer
   :members:
   :inherited-members:
   :show-inheritance:

..
   :exclude-members: from_config  ; remove this line or edit as you like

.. autoclass:: Seq2SeqDs
   :members:
   :show-inheritance:

..
   :exclude-members: from_config  ; remove this line or edit as you like

.. autoclass:: CharConfusionMatrix
   :members:
   :show-inheritance:


.. autoclass:: DemapperLSTM
   :members:
   :show-inheritance:


Functions
---------

.. autofunction:: char_similarity

.. autofunction:: fast_cer

.. autofunction:: fast_numpy_to_str

.. autofunction:: fast_str_to_numpy

.. autofunction:: print_err

.. autofunction:: extract_transcription_from_page_xml


Entry Points
------------

Functions that act as **entry points** (e.g., console scripts, `python -m pylelemmatize`, or a `cli` module). Point directly at the callable your packaging exposes.

.. autofunction:: pylelemmatize.main_infer_one2one

..
   .. note::

      If your console script is defined in `pyproject.toml` or `setup.cfg` as, e.g.,

      - ``pylemmatize = pylelemmatize.cli:main``

      then documenting ``cli.main`` (as above) will show exactly what users run.
