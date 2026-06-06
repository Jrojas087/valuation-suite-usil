// Course content and quiz data for Control Interno PYMES Paraguay
const COURSE = {
  title: "Control Interno para PYMES",
  subtitle: "Curso de Auto-instrucción",
  institution: "Programa de Capacitación Empresarial — Paraguay",
  passingScore: 70,

  modules: [
    {
      id: 1,
      title: "Fundamentos del Control Interno",
      shortTitle: "Fundamentos",
      duration: "25 min",
      icon: "📚",
      content: `
        <h2>¿Qué es el Control Interno?</h2>
        <p>El <strong>control interno</strong> es un proceso llevado a cabo por el directorio, la gerencia y demás personal de una organización, diseñado para proporcionar un grado de <em>seguridad razonable</em> en cuanto a la consecución de los siguientes objetivos:</p>
        <ul>
          <li><strong>Objetivos operativos:</strong> Eficacia y eficiencia en las operaciones.</li>
          <li><strong>Objetivos de información:</strong> Fiabilidad de la información financiera y no financiera.</li>
          <li><strong>Objetivos de cumplimiento:</strong> Acatamiento de leyes y normas aplicables.</li>
        </ul>
        <div class="highlight-box">
          <strong>Concepto clave:</strong> El control interno proporciona <em>seguridad razonable</em>, no absoluta. Ningún sistema puede garantizar el éxito total porque siempre existe el factor humano y circunstancias imprevistas.
        </div>
        <h2>El Marco COSO</h2>
        <p>El <strong>Marco Integrado de Control Interno COSO 2013</strong> (Committee of Sponsoring Organizations) es el estándar internacional más reconocido. Define <strong>cinco componentes interrelacionados</strong> y <strong>17 principios</strong> que soportan a cada componente:</p>
        <ol>
          <li>Entorno de Control</li>
          <li>Evaluación de Riesgos</li>
          <li>Actividades de Control</li>
          <li>Información y Comunicación</li>
          <li>Actividades de Supervisión</li>
        </ol>
        <h2>¿Por qué importa para las PYMES en Paraguay?</h2>
        <p>Las PYMES representan más del <strong>90 % del tejido empresarial paraguayo</strong>. Sin controles adecuados enfrentan:</p>
        <ul>
          <li>Pérdidas por fraude o errores no detectados</li>
          <li>Dificultades para acceder a crédito (BNF, AFD)</li>
          <li>Incumplimientos tributarios ante la SET</li>
          <li>Ineficiencias que reducen la competitividad</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Implementar control interno no requiere grandes inversiones. Comenzar con procedimientos escritos simples ya genera un impacto significativo en cualquier PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál es la finalidad principal del control interno según el Marco COSO?",
          options: [
            "Maximizar las ganancias de la empresa",
            "Proporcionar seguridad razonable en el logro de objetivos operativos, de información y de cumplimiento",
            "Eliminar completamente todos los riesgos del negocio",
            "Controlar exclusivamente las actividades del personal"
          ],
          correct: 1,
          explanation: "El control interno proporciona seguridad RAZONABLE —no absoluta— sobre tres categorías de objetivos: operativos, de información y de cumplimiento."
        },
        {
          question: "¿Cuántos componentes tiene el Marco Integrado COSO 2013?",
          options: ["3 componentes", "4 componentes", "5 componentes", "6 componentes"],
          correct: 2,
          explanation: "El Marco COSO 2013 define 5 componentes: Entorno de Control, Evaluación de Riesgos, Actividades de Control, Información y Comunicación, y Actividades de Supervisión."
        },
        {
          question: "¿El control interno puede garantizar que una empresa siempre logre sus objetivos?",
          options: [
            "Sí, si se implementa correctamente garantiza el éxito total",
            "No, solo proporciona seguridad razonable, no absoluta",
            "Sí, pero únicamente en empresas con más de 50 empleados",
            "Depende del sector económico de la empresa"
          ],
          correct: 1,
          explanation: "El control interno proporciona seguridad RAZONABLE, no absoluta. El factor humano y los imprevistos siempre representan un riesgo residual."
        }
      ]
    },
    {
      id: 2,
      title: "Entorno de Control",
      shortTitle: "Entorno de Control",
      duration: "25 min",
      icon: "🏢",
      content: `
        <h2>¿Qué es el Entorno de Control?</h2>
        <p>El <strong>Entorno de Control</strong> es el componente base de todos los demás. Establece el tono de la organización e influye en la conciencia de control del personal. Comprende los estándares, procesos y estructuras que sirven de base para implementar el control interno.</p>
        <h2>Elementos del Entorno de Control</h2>
        <ul>
          <li><strong>Integridad y valores éticos:</strong> Código de conducta que guía el comportamiento de todos en la organización.</li>
          <li><strong>Competencia del personal:</strong> Habilidades y conocimientos necesarios para cumplir las responsabilidades asignadas.</li>
          <li><strong>Filosofía y estilo operativo de la dirección:</strong> El "tono desde la cima" que marca cómo la gerencia afronta los riesgos y los controles.</li>
          <li><strong>Estructura organizacional:</strong> Distribución de autoridad y responsabilidad, con organigramas claros.</li>
          <li><strong>Políticas y prácticas de recursos humanos:</strong> Contratación, evaluación y capacitación del personal.</li>
        </ul>
        <div class="highlight-box">
          <strong>"El tono desde la cima":</strong> Cuando los dueños y gerentes demuestran con su propio comportamiento que la ética y los controles importan, el resto de la organización los sigue. Un líder que omite controles "por conveniencia" destruye la cultura de control.
        </div>
        <h2>En la práctica para una PYME</h2>
        <p>Incluso en una empresa de 5 empleados, el entorno de control existe:</p>
        <ul>
          <li>¿El dueño predica con el ejemplo en cuanto a honestidad?</li>
          <li>¿Existe un organigrama, aunque sea simple?</li>
          <li>¿Hay descripciones de puestos escritas?</li>
          <li>¿Se capacita al personal al ingresar?</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Redactar un breve "Código de Conducta" de una página y compartirlo con todo el equipo es un excelente primer paso para formalizar el Entorno de Control.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál componente COSO se considera la base de todos los demás?",
          options: [
            "Evaluación de Riesgos",
            "Entorno de Control",
            "Actividades de Control",
            "Actividades de Supervisión"
          ],
          correct: 1,
          explanation: "El Entorno de Control es el componente base. Sin un tono adecuado desde la cima, los demás componentes pierden efectividad."
        },
        {
          question: "¿Qué elemento NO forma parte del Entorno de Control?",
          options: [
            "Valores éticos e integridad",
            "Identificación y análisis de riesgos",
            "Estructura organizacional",
            "Políticas de recursos humanos"
          ],
          correct: 1,
          explanation: "La identificación y análisis de riesgos corresponde al componente 'Evaluación de Riesgos', no al Entorno de Control."
        },
        {
          question: "¿Por qué son importantes los valores éticos en el Entorno de Control?",
          options: [
            "Únicamente para cumplir con los requisitos de clientes externos",
            "Porque establecen el 'tono desde la cima' que guía el comportamiento de toda la organización",
            "Solo son relevantes para empresas que cotizan en bolsa",
            "Para reducir la carga impositiva ante la SET"
          ],
          correct: 1,
          explanation: "Los valores éticos definen el 'tono desde la cima'. Cuando la dirección actúa con integridad, genera una cultura de control que impregna toda la organización."
        }
      ]
    },
    {
      id: 3,
      title: "Evaluación de Riesgos",
      shortTitle: "Evaluación de Riesgos",
      duration: "30 min",
      icon: "⚠️",
      content: `
        <h2>¿Qué es la Evaluación de Riesgos?</h2>
        <p>La <strong>Evaluación de Riesgos</strong> implica identificar y analizar los riesgos relevantes para el logro de los objetivos de la organización, formando la base para determinar cómo deben gestionarse.</p>
        <h2>Riesgo Inherente vs. Riesgo Residual</h2>
        <ul>
          <li><strong>Riesgo inherente:</strong> El riesgo que existe antes de aplicar cualquier control. Es el riesgo "puro" del negocio.</li>
          <li><strong>Riesgo residual:</strong> El riesgo que permanece después de aplicar los controles. Debe estar dentro del nivel de tolerancia aceptado por la dirección.</li>
        </ul>
        <div class="highlight-box">
          <strong>Fórmula conceptual:</strong><br>
          Riesgo Residual = Riesgo Inherente − Efecto de los Controles<br><br>
          El objetivo es que el riesgo residual quede dentro del <em>apetito de riesgo</em> definido por la organización.
        </div>
        <h2>Tipos de Riesgos en una PYME</h2>
        <ul>
          <li><strong>Estratégicos:</strong> Cambios en el mercado, nueva competencia.</li>
          <li><strong>Operativos:</strong> Fallas en procesos, errores del personal.</li>
          <li><strong>Financieros:</strong> Falta de liquidez, morosidad de clientes.</li>
          <li><strong>De cumplimiento:</strong> Incumplimiento tributario, laboral o legal.</li>
          <li><strong>De fraude:</strong> Malversación de activos, manipulación de registros.</li>
        </ul>
        <h2>Respuestas al Riesgo</h2>
        <p>Ante un riesgo identificado, la organización puede elegir:</p>
        <ol>
          <li><strong>Aceptar:</strong> Asumir el riesgo sin acción adicional (riesgo bajo).</li>
          <li><strong>Evitar:</strong> Eliminar la actividad que genera el riesgo.</li>
          <li><strong>Reducir:</strong> Implementar controles para disminuir la probabilidad o impacto.</li>
          <li><strong>Compartir/Transferir:</strong> Contratar seguros, tercerizar actividades.</li>
        </ol>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Elaborar una matriz de riesgos simple (lista de riesgos × probabilidad × impacto) ayuda a priorizar los riesgos que merecen mayor atención en una PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Qué es el 'riesgo residual'?",
          options: [
            "El riesgo que prácticamente no existe en la organización",
            "El riesgo que permanece después de aplicar los controles internos",
            "El riesgo más grave identificado en el negocio",
            "Un tipo específico de riesgo financiero"
          ],
          correct: 1,
          explanation: "El riesgo residual es el que queda tras aplicar los controles. Debe estar dentro del nivel de tolerancia aceptado por la dirección."
        },
        {
          question: "¿Cuáles son las cuatro respuestas posibles ante un riesgo identificado?",
          options: [
            "Solo aceptarlo o eliminarlo completamente",
            "Ignorarlo o documentarlo para futuras revisiones",
            "Aceptar, evitar, reducir o compartir/transferir",
            "Solo transferirlo a un tercero mediante seguros"
          ],
          correct: 2,
          explanation: "Las cuatro respuestas al riesgo son: Aceptar, Evitar, Reducir y Compartir/Transferir. La elección depende del costo-beneficio de cada opción."
        },
        {
          question: "El COSO señala que existe un riesgo específico que siempre debe evaluarse. ¿Cuál es?",
          options: [
            "La intensidad de la competencia en el mercado",
            "Los cambios tecnológicos del sector",
            "El riesgo de fraude",
            "Las fluctuaciones del tipo de cambio"
          ],
          correct: 2,
          explanation: "COSO establece explícitamente que el riesgo de fraude debe considerarse siempre como parte del proceso de Evaluación de Riesgos, independientemente del tamaño de la empresa."
        }
      ]
    },
    {
      id: 4,
      title: "Actividades de Control",
      shortTitle: "Actividades de Control",
      duration: "30 min",
      icon: "🔧",
      content: `
        <h2>¿Qué son las Actividades de Control?</h2>
        <p>Las <strong>Actividades de Control</strong> son las políticas y procedimientos que ayudan a asegurar que se lleven a cabo las respuestas de la dirección ante los riesgos identificados. Ocurren en todos los niveles de la organización.</p>
        <h2>Tipos de Controles</h2>
        <h3>Según su naturaleza:</h3>
        <ul>
          <li><strong>Controles preventivos:</strong> Evitan que los errores o irregularidades ocurran. <em>Ejemplo: requerir dos firmas para pagos superiores a ₲ 500.000.</em></li>
          <li><strong>Controles detectivos:</strong> Identifican errores o irregularidades que ya ocurrieron. <em>Ejemplo: conciliación bancaria mensual.</em></li>
          <li><strong>Controles correctivos:</strong> Corrigen los problemas detectados. <em>Ejemplo: proceso de reversión de asientos incorrectos.</em></li>
        </ul>
        <h3>Según su forma de ejecución:</h3>
        <ul>
          <li><strong>Controles manuales:</strong> Realizados por personas. <em>Ejemplo: revisión física del inventario.</em></li>
          <li><strong>Controles automatizados:</strong> Realizados por sistemas informáticos. <em>Ejemplo: validación automática en el sistema contable.</em></li>
        </ul>
        <div class="highlight-box">
          <strong>Segregación de funciones:</strong> Es una de las actividades de control más importantes. Consiste en separar las responsabilidades de <strong>autorizar</strong>, <strong>registrar</strong> y <strong>custodiar</strong> activos en personas diferentes. Esto dificulta que una sola persona pueda cometer y ocultar un fraude.
        </div>
        <h2>Actividades de Control más comunes en PYMES</h2>
        <ul>
          <li>Aprobaciones y autorizaciones documentadas</li>
          <li>Conciliaciones bancarias mensuales</li>
          <li>Control de inventarios (conteos periódicos)</li>
          <li>Revisión de facturas antes del pago</li>
          <li>Controles de acceso a sistemas informáticos</li>
          <li>Custodia de activos (cajas, almacenes con llave)</li>
          <li>Archivo ordenado de documentos fuente</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> En PYMES pequeñas donde la segregación completa es difícil, el dueño/gerente puede compensar mediante supervisión directa y revisiones periódicas sorpresa.
        </div>
      `,
      quiz: [
        {
          question: "¿En qué consiste la segregación de funciones?",
          options: [
            "Dividir la empresa en múltiples departamentos independientes",
            "Separar las responsabilidades de autorizar, registrar y custodiar activos en personas distintas",
            "Contratar personal adicional para cada tarea",
            "Usar sistemas informáticos en lugar de controles manuales"
          ],
          correct: 1,
          explanation: "La segregación de funciones separa las responsabilidades de AUTORIZAR, REGISTRAR y CUSTODIAR en diferentes personas, dificultando que una sola cometa y oculte un fraude."
        },
        {
          question: "Un control que PREVIENE errores antes de que ocurran se denomina:",
          options: [
            "Control detectivo",
            "Control correctivo",
            "Control preventivo",
            "Control compensatorio"
          ],
          correct: 2,
          explanation: "Los controles PREVENTIVOS actúan antes de que ocurra el error o irregularidad. Los detectivos los identifican después y los correctivos los remedian."
        },
        {
          question: "¿Cuál de los siguientes es un ejemplo claro de actividad de control?",
          options: [
            "Fijar los objetivos estratégicos de la empresa",
            "Identificar los riesgos del negocio",
            "Realizar una conciliación bancaria mensual",
            "Comunicar los resultados anuales a los empleados"
          ],
          correct: 2,
          explanation: "La conciliación bancaria mensual es un control DETECTIVO que identifica discrepancias entre los registros contables y el extracto bancario."
        }
      ]
    },
    {
      id: 5,
      title: "Información y Comunicación",
      shortTitle: "Información y Comunicación",
      duration: "20 min",
      icon: "📡",
      content: `
        <h2>¿Por qué es clave la Información y Comunicación?</h2>
        <p>Para que el control interno funcione, todos en la organización necesitan información relevante y oportuna. La <strong>Información y Comunicación</strong> como componente COSO abarca tanto los sistemas que generan datos internos como los canales por los que fluye esa información.</p>
        <h2>Información de Calidad</h2>
        <p>La información útil para el control interno debe ser:</p>
        <ul>
          <li><strong>Relevante:</strong> Relacionada con las decisiones que deben tomarse.</li>
          <li><strong>Oportuna:</strong> Disponible en el momento en que se necesita.</li>
          <li><strong>Suficiente:</strong> Con el nivel de detalle adecuado.</li>
          <li><strong>Accesible:</strong> Disponible para quienes la necesitan.</li>
          <li><strong>Protegida:</strong> Con acceso restringido a personas autorizadas.</li>
        </ul>
        <h2>Comunicación Interna</h2>
        <p>La comunicación debe fluir en todas las direcciones:</p>
        <ul>
          <li><strong>Descendente (de arriba hacia abajo):</strong> La dirección comunica objetivos, políticas y responsabilidades.</li>
          <li><strong>Ascendente (de abajo hacia arriba):</strong> El personal reporta problemas, irregularidades y sugerencias.</li>
          <li><strong>Horizontal:</strong> Coordinación entre áreas o departamentos.</li>
        </ul>
        <div class="highlight-box">
          <strong>Canales de denuncia (Whistleblowing):</strong> Mecanismos que permiten al personal reportar irregularidades de manera confidencial y sin temor a represalias. Son una herramienta clave para detectar fraudes a tiempo.
        </div>
        <h2>Comunicación Externa</h2>
        <p>Las PYMES también deben gestionar la comunicación con:</p>
        <ul>
          <li>Clientes y proveedores</li>
          <li>Entidades reguladoras (SET, MTESS, municipalidades)</li>
          <li>Entidades financieras (bancos, cooperativas)</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Establecer una reunión semanal breve con el equipo para compartir indicadores clave es una forma simple y efectiva de mantener la comunicación interna activa en una PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Para qué sirven los canales de denuncia (whistleblowing) en el control interno?",
          options: [
            "Exclusivamente para atender quejas de clientes externos",
            "Para que los empleados reporten irregularidades de forma confidencial y sin represalias",
            "Para comunicar los resultados financieros a los inversionistas",
            "Para recibir órdenes de compra de proveedores"
          ],
          correct: 1,
          explanation: "Los canales de denuncia permiten al personal reportar irregularidades con confidencialidad y protección contra represalias, siendo clave para la detección temprana de fraudes."
        },
        {
          question: "La comunicación interna en una organización debe fluir:",
          options: [
            "Solo de la dirección hacia el personal (descendente)",
            "Solo entre personas del mismo nivel jerárquico (horizontal)",
            "En todas las direcciones: descendente, ascendente y horizontal",
            "Solo hacia el exterior de la organización"
          ],
          correct: 2,
          explanation: "Una comunicación efectiva fluye en las tres direcciones: descendente (dirección→personal), ascendente (personal→dirección) y horizontal (entre áreas)."
        },
        {
          question: "¿Qué características debe tener la información para ser útil en el control interno?",
          options: [
            "Ser lo más abundante y detallada posible en todo momento",
            "Ser relevante, oportuna, suficiente, accesible y protegida",
            "Ser exclusivamente de naturaleza financiera y contable",
            "Estar disponible solo para la alta dirección"
          ],
          correct: 1,
          explanation: "La información de calidad debe ser relevante, oportuna, suficiente, accesible y protegida. El exceso de información también puede ser un problema (ruido)."
        }
      ]
    },
    {
      id: 6,
      title: "Actividades de Supervisión",
      shortTitle: "Supervisión",
      duration: "20 min",
      icon: "🔍",
      content: `
        <h2>¿Qué son las Actividades de Supervisión?</h2>
        <p>Las <strong>Actividades de Supervisión</strong> (o Monitoreo) implican evaluar si los controles internos están presentes y funcionando correctamente. Es el componente que garantiza que el sistema de control interno evolucione y mejore con el tiempo.</p>
        <h2>Tipos de Evaluaciones</h2>
        <ul>
          <li><strong>Evaluaciones continuas:</strong> Integradas en las operaciones normales del negocio. <em>Ejemplo: el gerente revisa diariamente el resumen de ventas vs. cobros.</em></li>
          <li><strong>Evaluaciones periódicas independientes:</strong> Revisiones separadas realizadas por personas distintas a quienes operan el control. <em>Ejemplo: auditoría interna anual, revisión del contador externo.</em></li>
        </ul>
        <div class="highlight-box">
          <strong>La combinación ideal:</strong> Evaluaciones continuas para detectar problemas rápidamente + evaluaciones periódicas independientes para una visión objetiva. En PYMES pequeñas, el dueño puede realizar ambas.
        </div>
        <h2>Clasificación de Deficiencias</h2>
        <p>Cuando se identifican problemas en el control interno, se clasifican según su gravedad:</p>
        <ul>
          <li><strong>Deficiencia menor:</strong> Un control no funciona como se diseñó, pero hay otros controles que compensan.</li>
          <li><strong>Deficiencia significativa:</strong> Una deficiencia que merece atención, aunque no es tan grave como para considerarla material.</li>
          <li><strong>Debilidad material:</strong> Una deficiencia grave que podría resultar en errores o irregularidades significativas no detectadas a tiempo.</li>
        </ul>
        <h2>¿A quién se reportan las deficiencias?</h2>
        <ul>
          <li>Las deficiencias menores al responsable del área afectada.</li>
          <li>Las deficiencias significativas y debilidades materiales a la dirección y, en casos relevantes, al órgano de gobierno (directorio, socios).</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Una lista de verificación mensual con los principales controles (¿se hizo la conciliación? ¿se aprobaron los pagos con doble firma? ¿se revisó el inventario?) es una herramienta de supervisión continua ideal para PYMES.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál es la diferencia entre evaluaciones continuas y evaluaciones periódicas independientes?",
          options: [
            "No hay ninguna diferencia práctica entre ambas",
            "Las continuas son parte de las operaciones normales; las periódicas son revisiones separadas por personas distintas",
            "Las periódicas siempre son más importantes e informativas",
            "Las continuas solo pueden realizarlas auditores externos certificados"
          ],
          correct: 1,
          explanation: "Las evaluaciones CONTINUAS están integradas en la operación diaria. Las PERIÓDICAS son separadas y realizadas por personas distintas a quienes operan el control, lo que les da mayor objetividad."
        },
        {
          question: "¿A quién deben reportarse las deficiencias significativas y debilidades materiales del control interno?",
          options: [
            "Solo al contador que detectó el problema",
            "Al empleado que cometió el error",
            "A la dirección y, si corresponde, al órgano de gobierno (directorio, socios)",
            "Exclusivamente a los accionistas externos"
          ],
          correct: 2,
          explanation: "Las deficiencias significativas y debilidades materiales deben reportarse a la dirección y, en casos relevantes, al órgano de gobierno para que puedan tomar acciones correctivas."
        },
        {
          question: "¿Qué es una 'debilidad material' en el control interno?",
          options: [
            "Una deficiencia menor fácilmente corregible sin consecuencias",
            "La avería de un equipo o sistema informático",
            "Una deficiencia grave que podría resultar en errores o irregularidades significativas no detectados a tiempo",
            "La falta de tecnología moderna en la empresa"
          ],
          correct: 2,
          explanation: "Una DEBILIDAD MATERIAL es la deficiencia más grave. Implica que los errores o irregularidades significativas podrían no ser detectados oportunamente por el sistema de control."
        }
      ]
    }
  ],

  finalExam: [
    {
      question: "¿Cuál es el marco de referencia internacional más reconocido para el control interno?",
      options: ["ISO 9001", "COSO", "NIIF/IFRS", "Basilea III"],
      correct: 1,
      module: 1
    },
    {
      question: "¿Cuántos componentes tiene el Marco Integrado COSO 2013?",
      options: ["3", "4", "5", "7"],
      correct: 2,
      module: 1
    },
    {
      question: "El componente que establece las bases éticas y estructurales de todo el sistema de control interno es:",
      options: ["Evaluación de Riesgos", "Actividades de Control", "Entorno de Control", "Actividades de Supervisión"],
      correct: 2,
      module: 2
    },
    {
      question: "¿Qué significa que el control interno proporciona 'seguridad razonable'?",
      options: [
        "Garantiza el 100% de éxito en el logro de objetivos",
        "No existe ninguna garantía en absoluto",
        "Reduce pero no elimina los riesgos al logro de objetivos",
        "Solo aplica a empresas con más de 100 empleados"
      ],
      correct: 2,
      module: 1
    },
    {
      question: "La segregación de funciones es importante principalmente porque:",
      options: [
        "Reduce los costos de personal en la empresa",
        "Evita que una sola persona controle completamente una transacción, dificultando el fraude",
        "Mejora la productividad individual de cada empleado",
        "Es un requisito legal obligatorio para todas las empresas en Paraguay"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "¿Cuál es un ejemplo de control PREVENTIVO en una PYME?",
      options: [
        "Revisión de gastos al cierre del mes",
        "Auditoría anual por contador externo",
        "Requerir dos firmas para pagos superiores a un monto establecido",
        "Conciliación bancaria mensual"
      ],
      correct: 2,
      module: 4
    },
    {
      question: "¿Qué tipo de riesgo se asocia con el incumplimiento de leyes y regulaciones?",
      options: ["Riesgo operativo", "Riesgo financiero", "Riesgo de cumplimiento", "Riesgo estratégico"],
      correct: 2,
      module: 3
    },
    {
      question: "Cuando una PYME identifica un riesgo que no puede controlar completamente, ¿qué puede hacer?",
      options: [
        "Ignorarlo hasta que se manifieste",
        "Cerrar esa línea de negocio inmediatamente",
        "Aceptarlo, documentarlo o transferirlo (ej. mediante seguros)",
        "Contratar más personal para vigilarlo"
      ],
      correct: 2,
      module: 3
    },
    {
      question: "Las políticas y procedimientos escritos son importantes en el control interno porque:",
      options: [
        "Son exigidos directamente por la SET para todas las PYMES",
        "Aseguran que las actividades de control se ejecuten de forma consistente",
        "Aumentan automáticamente el valor de mercado de la empresa",
        "Solo son útiles en empresas con más de 50 empleados"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "¿Qué elemento del Entorno de Control se conoce como 'el tono desde la cima'?",
      options: [
        "Las políticas de contratación de RRHH",
        "Los sistemas de información contable",
        "La filosofía y el estilo operativo de la dirección",
        "Los manuales de procedimientos operativos"
      ],
      correct: 2,
      module: 2
    },
    {
      question: "Un canal de denuncia efectivo debe garantizar principalmente:",
      options: [
        "La identificación pública del denunciante ante toda la empresa",
        "Que todas las denuncias se archiven sin investigación previa",
        "Confidencialidad y protección contra represalias al denunciante",
        "Que solo los gerentes puedan reportar irregularidades"
      ],
      correct: 2,
      module: 5
    },
    {
      question: "¿Con qué frecuencia mínima se recomienda realizar la conciliación bancaria en una PYME?",
      options: [
        "Una vez al año, al cierre del ejercicio",
        "Solo cuando se detectan diferencias en los saldos",
        "Al menos mensualmente",
        "Cada cinco años con el contador externo"
      ],
      correct: 2,
      module: 4
    },
    {
      question: "El riesgo de fraude debe considerarse:",
      options: [
        "Solo en la evaluación anual del contador externo",
        "Como parte del proceso de Evaluación de Riesgos de forma permanente",
        "Únicamente si ya existen sospechas concretas de irregularidades",
        "Solo en empresas grandes con muchos empleados"
      ],
      correct: 1,
      module: 3
    },
    {
      question: "¿Qué son los 17 principios del Marco COSO 2013?",
      options: [
        "Normas contables obligatorias establecidas por la SET",
        "Conceptos fundamentales que soportan y desarrollan los 5 componentes del control interno",
        "Leyes paraguayas específicas sobre control interno empresarial",
        "Indicadores financieros para medir la rentabilidad"
      ],
      correct: 1,
      module: 1
    },
    {
      question: "¿Cuál es el objetivo principal de las evaluaciones de supervisión continua?",
      options: [
        "Reemplazar completamente la auditoría externa anual",
        "Detectar y corregir deficiencias del control interno de manera oportuna",
        "Controlar el cumplimiento del horario de trabajo del personal",
        "Preparar los estados financieros del ejercicio"
      ],
      correct: 1,
      module: 6
    },
    {
      question: "En una PYME pequeña donde es difícil segregar todas las funciones, ¿cómo se pueden compensar esas limitaciones?",
      options: [
        "Es imposible tener control interno efectivo en empresas tan pequeñas",
        "Mediante controles compensatorios como supervisión directa del dueño o revisiones periódicas sorpresa",
        "Contratando siempre auditores externos certificados de forma mensual",
        "Ignorando ese componente hasta que la empresa crezca"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "¿Cuál de los siguientes NO es un objetivo del control interno según el Marco COSO?",
      options: [
        "Objetivos de operaciones (eficiencia y eficacia)",
        "Objetivos de reporte (fiabilidad de la información)",
        "Objetivos de cumplimiento (leyes y regulaciones)",
        "Objetivos de maximización de ganancias"
      ],
      correct: 3,
      module: 1
    },
    {
      question: "Las políticas de contratación, evaluación y capacitación del personal forman parte de:",
      options: [
        "Actividades de Control",
        "Entorno de Control",
        "Evaluación de Riesgos",
        "Actividades de Supervisión"
      ],
      correct: 1,
      module: 2
    },
    {
      question: "Una 'deficiencia significativa' en el control interno:",
      options: [
        "Se puede ignorar si el negocio muestra buenas ganancias",
        "Merece atención de la dirección aunque no sea tan grave como una debilidad material",
        "Solo es relevante en empresas que cotizan en bolsa",
        "Se corrige automáticamente con el paso del tiempo"
      ],
      correct: 1,
      module: 6
    },
    {
      question: "¿Por qué es especialmente importante implementar control interno en las PYMES paraguayas?",
      options: [
        "Porque todas las PYMES en Paraguay están legalmente obligadas a tener un sistema COSO",
        "Porque mejora la eficiencia, previene fraudes y facilita el acceso a financiamiento",
        "Exclusivamente porque reduce la carga impositiva ante la SET",
        "No es realmente importante para PYMES pequeñas"
      ],
      correct: 1,
      module: 1
    }
  ]
};
